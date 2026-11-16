import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcribe_youtube as app


def word(text, start, end, speaker):
    return {'punctuated_word': text, 'start': start, 'end': end, 'speaker': speaker}


def sample():
    return {'results': {'utterances': [{'words': [word('가격은?', 0, 1, 0),
        word('좋아요.', 1.1, 2, 1), word('정말요.', 2.1, 3, 1)]}]}}


class TranscriptionTests(unittest.TestCase):
    def test_url_normalization(self):
        expected = 'https://www.youtube.com/watch?v=abcdefghijk'
        for url in ['https://youtu.be/abcdefghijk?t=3',
                    expected + '&list=playlist',
                    'https://www.youtube.com/shorts/abcdefghijk',
                    'https://youtube.com/live/abcdefghijk']:
            self.assertEqual(app.youtube_url(url), expected)
        for url in ['https://youtube.com.evil.com/watch?v=abcdefghijk',
                    'file:///tmp/a', 'https://youtube.com/playlist?list=test']:
            with self.assertRaises(ValueError):
                app.youtube_url(url)

    def test_mixed_speaker_utterance_is_split(self):
        rows = app.make_rows(sample(), {'SPEAKER_00': '진행자'})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['speaker_name'], '진행자')
        self.assertEqual(rows[1]['text'], '좋아요. 정말요.')
        self.assertEqual(rows[1]['start_seconds'], 1.1)

    def test_words_fallback_and_unknown(self):
        payload = {'results': {'channels': [{'alternatives': [{'words': [
            word('안녕', 0, 1, None), word('다시', 4, 5, None)]}]}]}}
        rows = app.make_rows(payload)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['speaker_id'] == 'UNKNOWN' for r in rows))

    def test_timestamp_carry(self):
        self.assertEqual(app.timestamp(59.9996, True), '00:01:00,000')
        self.assertEqual(app.timestamp(3600.125), '01:00:00.125')

    def test_export_unicode_and_formula_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            payload = {'results': {'utterances': [{'start': 0, 'end': 1,
                       'speaker': 0, 'transcript': '=위험한수식'}]}}
            app.save_outputs(folder, payload, {'title': '한국어'})
            csv = (folder / 'utterances.csv').read_text(encoding='utf-8-sig')
            self.assertIn("'=위험한수식", csv)
            rows = json.loads((folder / 'utterances.json').read_text(encoding='utf-8'))
            self.assertEqual(rows[0]['text'], '=위험한수식')
            self.assertIn('00:00:00,000 --> 00:00:01,000',
                          (folder / 'subtitles.srt').read_text(encoding='utf-8'))

    def test_no_speech_keeps_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(ValueError):
                app.save_outputs(folder, {'results': {}}, {})
            self.assertTrue((folder / 'deepgram_raw.json').exists())

    def test_api_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'audio.wav'
            audio.write_bytes(b'RIFFfake')
            with patch.object(app, 'urlopen', return_value=io.BytesIO(json.dumps(sample()).encode())) as call:
                self.assertEqual(app.transcribe(audio, 'TEST_SECRET', 'nova-3', 'ko'), sample())
            request = call.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'), 'Token TEST_SECRET')
            self.assertIn('diarize_model=latest', request.full_url)
            self.assertIn('language=ko', request.full_url)
            self.assertEqual(request.data, b'RIFFfake')

    def test_error_does_not_disclose_key(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'a.wav'
            audio.write_bytes(b'fake')
            error = HTTPError('https://api.deepgram.com', 401, 'TEST_SECRET', {}, None)
            with patch.object(app, 'urlopen', side_effect=error) as call:
                with self.assertRaises(RuntimeError) as caught:
                    app.transcribe(audio, 'TEST_SECRET', 'nova-3', 'ko')
                self.assertNotIn('TEST_SECRET', str(caught.exception))
                self.assertEqual(call.call_count, 1)

    def test_reexport_without_key(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / 'raw.json'
            raw.write_text(json.dumps(sample()), encoding='utf-8')
            with patch.object(app, 'transcribe', side_effect=AssertionError('API must not be called')):
                self.assertEqual(app.main(['--response', str(raw), '--output-dir', str(folder / 'out')]), 0)
            self.assertEqual(len(list((folder / 'out').glob('*/utterances.csv'))), 1)

    def test_setup_preserves_existing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / '.env.example').write_text('DEEPGRAM_API_KEY=placeholder', encoding='utf-8')
            with patch.object(app, 'BASE', folder):
                self.assertEqual(app.main(['--setup']), 0)
                (folder / '.env').write_text('DEEPGRAM_API_KEY=TEST_SECRET', encoding='utf-8')
                self.assertEqual(app.main(['--setup']), 0)
                self.assertIn('TEST_SECRET', (folder / '.env').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
