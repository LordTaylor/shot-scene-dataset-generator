import { app } from "../../scripts/app.js";

const MAX_SLOTS = 8;
const NODE_TYPE = "MultiLoraLoader";

function slotWidgets(node, idx) {
    return [
        node.widgets?.find(w => w.name === `lora_${idx}`),
        node.widgets?.find(w => w.name === `strength_${idx}`),
        node.widgets?.find(w => w.name === `enabled_${idx}`),
    ].filter(Boolean);
}

function refreshSlots(node) {
    for (let i = 1; i <= MAX_SLOTS; i++) {
        const visible = i <= node._activeSlots;
        for (const w of slotWidgets(node, i)) {
            w.hidden = !visible;
        }
    }
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, false);
}

function activeSlotCount(node) {
    // Find the last slot that has a non-None lora selected
    for (let i = MAX_SLOTS; i >= 1; i--) {
        const w = node.widgets?.find(w => w.name === `lora_${i}`);
        if (w && w.value && w.value !== "None") return i;
    }
    return 1;
}

app.registerExtension({
    name: "ShotScene.MultiLoraLoader",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this._activeSlots = 1;
            refreshSlots(this);

            this.addWidget("button", "＋ Add LoRA", null, () => {
                if (this._activeSlots < MAX_SLOTS) {
                    this._activeSlots++;
                    refreshSlots(this);
                }
            });

            this.addWidget("button", "− Remove LoRA", null, () => {
                if (this._activeSlots > 1) {
                    const ws = slotWidgets(this, this._activeSlots);
                    if (ws[0]) ws[0].value = "None";
                    if (ws[1]) ws[1].value = 1.0;
                    if (ws[2]) ws[2].value = true;
                    this._activeSlots--;
                    refreshSlots(this);
                }
            });
        };

        // Restore visible slot count when loading a saved workflow
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config) {
            onConfigure?.apply(this, arguments);
            this._activeSlots = activeSlotCount(this);
            refreshSlots(this);
        };
    },
});
