import json
import sys

import torch
import megatron
import mcore_bridge
from swift.megatron import megatron_sft_main
from swift.megatron.arguments import MegatronSftArguments


print(
    json.dumps(
        {
            "status": "IMPORT_SMOKE_OK",
            "python": sys.executable,
            "torch": torch.__version__,
            "megatron": getattr(megatron, "__file__", None),
            "mcore_bridge": mcore_bridge.__file__,
            "swift_arguments": MegatronSftArguments.__module__,
            "swift_entry": megatron_sft_main.__module__,
        },
        ensure_ascii=False,
    ),
    flush=True,
)
