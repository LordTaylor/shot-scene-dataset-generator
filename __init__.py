WEB_DIRECTORY = "./web"

try:
    from .nodes import UltraWildcardNode, WildcardAppend, MultiLoraLoader
except ImportError:
    from nodes import UltraWildcardNode, WildcardAppend, MultiLoraLoader

NODE_CLASS_MAPPINGS = {
    "UltraWildcardNode": UltraWildcardNode,
    "WildcardAppend":    WildcardAppend,
    "MultiLoraLoader":   MultiLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraWildcardNode": "🎲 Ultra Wildcard Node",
    "WildcardAppend":    "➕ Wildcard Append",
    "MultiLoraLoader":   "🔗 Multi LoRA Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
