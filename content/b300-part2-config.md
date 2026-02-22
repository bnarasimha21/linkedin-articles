# B300 Deep Dive - Part 2: Full System Configuration

## Status: Waiting for hardware availability

## Reference Configuration

**Server:** Dell PowerEdge XE9780

**GPUs:** 8x NVIDIA B300

**CPUs:** Intel Xeon

**Networking:**
- **Intra-node:** NVLink (GPU-to-GPU within server)
- **Inter-node:** 800 Gbps RoCEv2 (server-to-server)

## Content Plan

### Extension to existing B300 content:
1. Real-world inference/training benchmarks on this setup
2. NVLink scaling demos (8-GPU collective operations)
3. 800G RoCEv2 multi-node performance metrics
4. Practical examples: "What you can run on this config"

### Formats:
- [ ] LinkedIn article (Part 2)
- [ ] Remotion video (Part 2)
- [ ] X post thread

## Related Files
- B300 PDF: already created and merged
- B300 Remotion: in progress

---
*Created: 2026-02-22*
*Trigger: Narsi will ping when hardware is available*
