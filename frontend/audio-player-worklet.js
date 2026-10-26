"use strict";

class PCMOutputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(0);
    this.position = 0;
    this.sourceRate = 24_000;
    this.ended = false;
    this.drainedSent = false;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "start") {
        this.samples = new Float32Array(0);
        this.position = 0;
        this.sourceRate = Number(message.sampleRate) || 24_000;
        this.ended = false;
        this.drainedSent = false;
      } else if (message.type === "pcm" && message.samples instanceof Float32Array) {
        const merged = new Float32Array(this.samples.length + message.samples.length);
        merged.set(this.samples);
        merged.set(message.samples, this.samples.length);
        this.samples = merged;
      } else if (message.type === "end") {
        this.ended = true;
      } else if (message.type === "stop") {
        this.samples = new Float32Array(0);
        this.position = 0;
        this.ended = true;
        this.drainedSent = true;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0]?.[0];
    if (!output) return true;
    output.fill(0);
    const step = this.sourceRate / sampleRate;
    for (let index = 0; index < output.length; index += 1) {
      if (this.samples.length < 2 || this.position >= this.samples.length - 1) break;
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      output[index] = this.samples[left] * (1 - fraction) + this.samples[left + 1] * fraction;
      this.position += step;
    }

    const consumed = Math.min(Math.floor(this.position), Math.max(0, this.samples.length - 1));
    if (consumed > 0) {
      this.samples = this.samples.slice(consumed);
      this.position -= consumed;
    }
    if (this.ended && this.samples.length <= 1) {
      this.samples = new Float32Array(0);
      if (!this.drainedSent) {
        this.drainedSent = true;
        this.port.postMessage({ type: "drained" });
      }
    }
    return true;
  }
}

registerProcessor("pcm-output", PCMOutputProcessor);
