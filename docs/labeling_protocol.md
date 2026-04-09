# Malon Labeling Protocol

This document describes the image labeling guidelines used to construct the Malon training dataset. It is provided to support replication of the labeling methodology and extension of the dataset to new image sources. Images were labeled using a custom keyboard-driven annotation interface that displayed each image and recorded class assignments via hotkey. The annotation tool is not released as it was implemented for a specific local environment; the labeling criteria documented here are sufficient to replicate the process with any image viewer capable of sorting files by keypress.

---

## Quick Reference

| Class | Label | Criteria |
|---|---|---|
| 0 | Not useful | No extractable botanical features for CV |
| 1 | Atypical | Features present but non-standard preparation |
| 2 | Typical | Standard pressed specimen, suitable for CV |

**Decision test:** Would a CV algorithm be able to extract meaningful botanical features from this image?
- Class 0: No
- Class 1: Yes, but limited or atypical
- Class 2: Yes, standard case

**Boundary rule:** When uncertain, prefer the lower class.

---

## Detailed Class Definitions

### Class 0: Not Useful

Include:
- Field photographs (iNaturalist-style, natural backgrounds)
- No visible plant structures (label-only scans, blank sheets)
- Botanical illustrations and drawings (no preserved material)
- Closed seed packets and fragment envelopes (contents not visible)
- Plant material too fragmentary or degraded for meaningful analysis
- Images where scan artifacts prevent interpretation

Examples:
- iNaturalist observation photos in natural settings
- Herbarium sheet showing only collection labels
- Botanical illustrations with handwritten annotations
- Unopened seed packages
- Lemma fragments reduced to unidentifiable debris

### Class 1: Atypical

Include:
- Wood cross-sections, bark samples, stem segments
- Open fragment packets with visible plant material
- Seed collections where individual seeds are visible
- Small aquatics or Bryophytes with identifiable morphological structures
- Processed plant products retaining some biological structure

Examples:
- Wood sections showing anatomical features
- Opened fragment envelopes with visible leaf pieces
- *Spirodela* specimens with intact fronds
- Opened packages showing seeds or other identifiable material

### Class 2: Typical

Include:
- Traditional pressed and mounted specimens
- Clear morphological features visible (leaves, flowers, stems)
- Standard herbarium sheet format
- Material suitable for comprehensive morphological analysis
- Well-preserved specimens enabling detailed trait extraction

Examples:
- Standard vascular plant specimens with leaves and flowers
- Well-mounted fern specimens showing frond detail
- Complete specimens with reproductive structures
- *Lemna* specimens with identifiable individual fronds

---

## Edge Case Decision Rules

**Institutional context vs. CV utility:**
CV utility takes precedence. A legitimate herbarium record may still be Class 0 if it is not useful for CV applications.

**Closed vs. open containers:**
- Closed = Class 0 (no visible features)
- Open = Class 1 if contents are identifiable

**Very small specimens:**
- Class 2 if morphological features are distinguishable
- Class 0 if reduced to unidentifiable fragments
- Class 1 if identifiable but challenging for analysis

**Quality threshold:**
Can basic plant structures be recognized and potentially measured? If unsure, choose the more conservative (lower) class.
