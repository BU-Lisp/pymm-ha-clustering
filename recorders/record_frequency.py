# SYSTEM IMPORTS
from telnetlib import Telnet
from typing import Dict, List, Tuple, Union
import argparse
import numpy as np
import os
import sys
import time


# PYTHON PROJECT IMPORTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=str, help="where to write files to")
    parser.add_argument("--t", type=float, default=1, help="periodicity (in seconds)")
    parser.add_argument("--port", type=int, default=4444, help="port of qemu telnet server")
    args = parser.parse_args()

    snapshot_msg: str = '{ "execute": "dump-guest-memory", "arguments": { "protocol": "file:%s.dump", "paging" : false } }'

    filenum: int = 0
    with Telnet("localhost", args.port) as tn:
        while True:
            try:
                current_filepath: str = os.path.join(args.out_dir, filenum)
                msg: str = snapshot_msg %s current_filepath
                tn.write(str.encode(msg))

                filenum += 1
            except KeyboardInterrupt:
                print("CTRL+C pressed...goodbye")
                return


if __name__ == "__main__":
    main()



# dr ronald lonpree(?) 978-462-8160

