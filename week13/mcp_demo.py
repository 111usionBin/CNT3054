import argparse
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--id", default="U0006")
    parser.add_argument("--speaker", default="SPEAKER_00")
    args = parser.parse_args()
    server = Path(__file__).with_name("mcp_server.py")
    params = StdioServerParameters(command=sys.executable, args=[str(server), str(args.result.resolve())])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            print("사용 가능한 도구:", ", ".join(t.name for t in (await session.list_tools()).tools))
            for name, values in [("find_disagreements", {"task": "stage"}),
                                 ("lookup_utterance", {"utterance_id": args.id}),
                                 ("speaker_emotions", {"model": "gpt", "speaker": args.speaker})]:
                result = await session.call_tool(name, values)
                if result.isError:
                    raise RuntimeError(result.content)
                print(name, *(c.text for c in result.content if c.type == "text"), sep="\n")


if __name__ == "__main__":
    asyncio.run(main())
