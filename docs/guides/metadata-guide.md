# DeepStream Metadata — Complete Guide

> **Goal:** Understand what DeepStream metadata is, what structs it contains,
> how data flows through them, and how to read/write them in C and Python.

---

## Table of Contents

1. [What Is Metadata in DeepStream?](#1-what-is-metadata-in-deepstream)
2. [The Metadata Hierarchy (Big Picture)](#2-the-metadata-hierarchy-big-picture)
3. [Core Structs — Full Reference](#3-core-structs--full-reference)
   - [NvDsBatchMeta](#nvdsbatchmeta)
   - [NvDsFrameMeta](#nvdsframemeta)
   - [NvDsObjectMeta](#nvdsobjectmeta)
   - [NvDsClassifierMeta](#nvdsclassifiermeta)
   - [NvDsLabelInfo](#nvdslabelinfo)
   - [NvDsUserMeta](#nvdsusermeta)
   - [NvDsDisplayMeta](#nvdsdisplaymeta)
4. [Bounding Box Structs](#4-bounding-box-structs)
5. [How Data Flows Through the Pipeline](#5-how-data-flows-through-the-pipeline)
6. [How to Access Metadata — C](#6-how-to-access-metadata--c)
7. [How to Access Metadata — Python](#7-how-to-access-metadata--python)
8. [Adding Custom / User Metadata](#8-adding-custom--user-metadata)
9. [Metadata Before nvstreammux](#9-metadata-before-nvstreammux)
10. [Key Header Files](#10-key-header-files)
11. [Important Rules & Gotchas](#11-important-rules--gotchas)

---

## 1. What Is Metadata in DeepStream?

Every video frame that travels through a GStreamer pipeline is wrapped in a
**GstBuffer**. DeepStream attaches its own metadata object — `NvDsBatchMeta` —
to that buffer. Instead of copying raw pixel data between plugins, plugins read
and write structured C structs that describe:

- Which stream a frame came from
- What objects were detected (bounding boxes, class IDs, confidence scores)
- Tracker IDs assigned to each object
- Secondary classifier labels (make, type, color, …)
- Any custom user data you want to carry

This metadata travels with the buffer through the entire pipeline so every
downstream plugin can read or extend it without touching the pixel data.

---

## 2. The Metadata Hierarchy (Big Picture)

```
GstBuffer
└── NvDsBatchMeta                    ← one per batch (created by nvstreammux)
    ├── batch_user_meta_list         ← custom data at batch level
    └── frame_meta_list              ← list of NvDsFrameMeta (one per frame)
        └── NvDsFrameMeta
            ├── frame_user_meta_list ← custom data at frame level
            ├── display_meta_list    ← shapes/text drawn by nvdsosd
            └── obj_meta_list        ← list of NvDsObjectMeta (one per object)
                └── NvDsObjectMeta
                    ├── detector_bbox_info   ← raw detector box
                    ├── tracker_bbox_info    ← tracker-adjusted box
                    ├── rect_params          ← final box used for OSD
                    ├── mask_params          ← segmentation mask (optional)
                    ├── obj_user_meta_list   ← custom data at object level
                    └── classifier_meta_list ← list of NvDsClassifierMeta
                        └── NvDsClassifierMeta
                            └── label_info_list ← list of NvDsLabelInfo
                                └── NvDsLabelInfo
```

The lists (`frame_meta_list`, `obj_meta_list`, …) are **GLib doubly-linked
lists** (`NvDsMetaList` = `GList`). Iterate them with `l = l->next`.

---

## 3. Core Structs — Full Reference

### NvDsBatchMeta

**Header:** `sources/include/nvdsmeta.h`  
**Created by:** `nvstreammux`  
**Extracted with:** `gst_buffer_get_nvds_batch_meta(buf)`

```c
typedef struct _NvDsBatchMeta {
    NvDsMetaPool  *frame_meta_pool;       // internal pool — don't use directly
    NvDsMetaPool  *obj_meta_pool;
    NvDsMetaPool  *classifier_meta_pool;
    NvDsMetaPool  *display_meta_pool;
    NvDsMetaPool  *user_meta_pool;
    NvDsMetaPool  *label_info_meta_pool;

    NvDsMetaList  *frame_meta_list;       // ← iterate this for frames
    NvDsMetaList  *batch_user_meta_list;  // ← batch-level custom data

    guint64        max_frames_in_batch;   // max frames per batch
    guint64        num_frames_in_batch;   // actual frames this batch

    NvDsMeta       base_meta;             // GStreamer meta base
} NvDsBatchMeta;
```

**Key fields you use:**

| Field | Type | Meaning |
|---|---|---|
| `frame_meta_list` | `NvDsMetaList *` | Linked list — walk this to get each frame |
| `num_frames_in_batch` | `guint64` | How many frames are in this batch |
| `batch_user_meta_list` | `NvDsMetaList *` | Attach batch-level custom data here |

---

### NvDsFrameMeta

**Header:** `sources/include/nvdsmeta.h`  
One node per video frame inside the batch.

```c
typedef struct _NvDsFrameMeta {
    NvDsBaseMeta  base_meta;

    guint   pad_index;          // which nvstreammux input pad this frame came from
    guint   batch_id;           // index of this frame in the batch (0..N-1)
    gint    frame_num;          // sequential frame number from the source
    guint64 buf_pts;            // buffer PTS (presentation timestamp) in ns
    guint64 ntp_timestamp;      // NTP wall-clock timestamp (ns)
    guint   source_id;          // source stream ID (same as pad_index usually)
    gint    num_surfaces_per_frame; // usually 1; >1 for 360° dewarped surfaces

    guint   num_obj_meta_in_use;   // number of objects detected in this frame
    guint   num_obj_meta_allocated;

    NvDsMetaList *obj_meta_list;          // ← iterate for objects
    NvDsMetaList *frame_user_meta_list;   // ← frame-level custom data
    NvDsMetaList *display_meta_list;      // ← OSD drawing commands

    gint    pipeline_width;     // frame width as it enters the pipeline
    gint    pipeline_height;

    gboolean bInferDone;        // TRUE after at least one inference ran
    guint    surface_type;      // surface memory type
    guint    surface_index;

    NvDsComp_BboxInfo  detector_bbox_info;
    NvDsComp_BboxInfo  tracker_bbox_info;
} NvDsFrameMeta;
```

**Key fields:**

| Field | Meaning |
|---|---|
| `source_id` | Which camera / stream this frame belongs to |
| `frame_num` | Sequential frame counter from that source |
| `buf_pts` | GStreamer timestamp (nanoseconds) |
| `ntp_timestamp` | Wall-clock time (nanoseconds), useful for logging events |
| `obj_meta_list` | Walk this to get every detected object |
| `bInferDone` | Whether inference has run on this frame |

---

### NvDsObjectMeta

**Header:** `sources/include/nvdsmeta.h`  
One node per detected object in a frame.

```c
typedef struct _NvDsObjectMeta {
    NvDsBaseMeta  base_meta;

    NvDsFrameMeta *frame_meta;          // back-pointer to the parent frame

    gint    class_id;                   // class index from detector model
    guint64 object_id;                  // tracker ID (UNTRACKED_OBJECT_ID if no tracker)
    gfloat  confidence;                 // detector confidence [0.0 – 1.0]
    gfloat  tracker_confidence;         // tracker confidence (NvDCF) or 1.0

    NvDsComp_BboxInfo detector_bbox_info;  // raw bbox from detector
    NvDsComp_BboxInfo tracker_bbox_info;   // bbox from tracker

    NvOSD_RectParams  rect_params;      // final bbox used for drawing (clipped)
    NvOSD_MaskParams  mask_params;      // segmentation mask (optional)
    NvOSD_TextParams  text_params;      // label text drawn by nvdsosd
    NvOSD_Arrow_Params arrow_params;    // (rarely used)

    gchar   obj_label[MAX_LABEL_SIZE];  // human-readable class label string

    NvDsMetaList *classifier_meta_list; // ← secondary classifier results
    NvDsMetaList *obj_user_meta_list;   // ← object-level custom data

    gint    unique_component_id;        // which nvinfer component detected this
} NvDsObjectMeta;
```

**Key fields:**

| Field | Meaning |
|---|---|
| `class_id` | Integer class index (e.g. 0=car, 1=person) |
| `object_id` | Persistent tracker ID across frames |
| `confidence` | Detector score |
| `rect_params` | Use this for the bounding box coordinates |
| `obj_label` | Class name string |
| `classifier_meta_list` | Walk this for secondary classifier results |

---

### NvDsClassifierMeta

**Header:** `sources/include/nvdsmeta.h`  
Attached per secondary classifier that ran on an object.

```c
typedef struct _NvDsClassifierMeta {
    NvDsBaseMeta  base_meta;

    guint            num_labels;           // number of output labels
    gint             unique_component_id;  // which nvinfer did this classification
    NvDsMetaList    *label_info_list;      // ← list of NvDsLabelInfo
} NvDsClassifierMeta;
```

---

### NvDsLabelInfo

The actual classification result for one label index.

```c
typedef struct _NvDsLabelInfo {
    NvDsBaseMeta  base_meta;

    guint   num_classes;        // total classes in this label
    gchar   result_label[MAX_LABEL_SIZE]; // winning class name
    guint   result_class_id;    // winning class index
    gfloat  result_prob;        // confidence of the winning class
    gchar  *pResult_label;      // pointer to result_label (same string)
    gint    label_id;           // label index within the classifier
} NvDsLabelInfo;
```

---

### NvDsUserMeta

Used for any custom/user-defined metadata at batch, frame, or object level.

```c
typedef struct _NvDsUserMeta {
    NvDsBaseMeta   base_meta;

    gpointer       user_meta_data;    // pointer to your custom struct
    NvDsMetaType   meta_type;         // your own type enum value
    gpointer       copy_func;         // called when meta is copied
    gpointer       release_func;      // called when meta is freed
} NvDsUserMeta;
```

---

### NvDsDisplayMeta

Commands sent to `nvdsosd` to draw shapes and text on screen.

```c
typedef struct _NvDsDisplayMeta {
    NvDsBaseMeta  base_meta;

    guint  num_rects;      // how many rectangles to draw
    guint  num_labels;     // how many text labels to draw
    guint  num_lines;      // how many lines to draw
    guint  num_arrows;
    guint  num_circles;

    NvOSD_RectParams  rect_params[MAX_ELEMENTS_IN_DISPLAY_META];
    NvOSD_TextParams  text_params[MAX_ELEMENTS_IN_DISPLAY_META];
    NvOSD_LineParams  line_params[MAX_ELEMENTS_IN_DISPLAY_META];
    NvOSD_ArrowParams arrow_params[MAX_ELEMENTS_IN_DISPLAY_META];
    NvOSD_CircleParams circle_params[MAX_ELEMENTS_IN_DISPLAY_META];

    gint    misc_osd_data[MAX_USER_FIELDS];
    gdouble reserved[MAX_RESERVED_FIELDS];
} NvDsDisplayMeta;
```

---

## 4. Bounding Box Structs

### NvDsComp_BboxInfo

Container that holds one component's bbox coordinates.

```c
typedef struct {
    NvOSD_RectParams org_bbox_coords; // left, top, width, height (pixels, float)
} NvDsComp_BboxInfo;
```

### NvOSD_RectParams

The actual rectangle used throughout DeepStream:

```c
typedef struct {
    float left;        // x coordinate of left edge (pixels)
    float top;         // y coordinate of top edge (pixels)
    float width;       // width in pixels
    float height;      // height in pixels
    guint border_width;
    NvOSD_ColorParams border_color;  // RGBA
    guint has_bg_color;
    NvOSD_ColorParams bg_color;
    gint  has_color_info;
    gint  color_id;
} NvOSD_RectParams;
```

### Three Bounding Boxes on NvDsObjectMeta

| Field | Who Sets It | What It Holds |
|---|---|---|
| `detector_bbox_info` | `nvinfer` | Raw detector output — may go outside frame |
| `tracker_bbox_info` | `nvtracker` | Tracker-smoothed box |
| `rect_params` | Last module to touch the object | Clipped to frame boundary — **use this for drawing** |

---

## 5. How Data Flows Through the Pipeline

```
Camera / File
    ↓
[decode]
    ↓
[nvstreammux]  ← creates NvDsBatchMeta, adds NvDsFrameMeta per frame
    ↓
[nvinfer]      ← adds NvDsObjectMeta to each NvDsFrameMeta.obj_meta_list
                  fills: class_id, confidence, detector_bbox_info, obj_label
    ↓
[nvtracker]    ← assigns object_id, fills tracker_bbox_info, tracker_confidence
    ↓
[nvinfer (secondary)] ← adds NvDsClassifierMeta to each NvDsObjectMeta
    ↓
[nvdsosd]      ← reads rect_params / text_params to draw on screen
    ↓
[your probe / plugin] ← read everything above
    ↓
[sink]
```

---

## 6. How to Access Metadata — C

### Minimal pad probe (the standard pattern)

```c
#include "gstnvdsmeta.h"

static GstPadProbeReturn
my_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);

    /* Step 1: get the batch meta from the buffer */
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    /* Step 2: walk each frame */
    for (NvDsMetaList *fl = batch_meta->frame_meta_list; fl; fl = fl->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fl->data;

        g_print("source_id=%u  frame_num=%d  pts=%" G_GUINT64_FORMAT "\n",
                frame_meta->source_id,
                frame_meta->frame_num,
                frame_meta->buf_pts);

        /* Step 3: walk each detected object */
        for (NvDsMetaList *ol = frame_meta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;

            g_print("  class_id=%d  label=%s  track_id=%" G_GUINT64_FORMAT
                    "  conf=%.2f\n",
                    obj->class_id, obj->obj_label,
                    obj->object_id, obj->confidence);

            /* Bounding box (use rect_params — it is clipped to frame) */
            NvOSD_RectParams *r = &obj->rect_params;
            g_print("  bbox: left=%.1f top=%.1f w=%.1f h=%.1f\n",
                    r->left, r->top, r->width, r->height);

            /* Step 4: secondary classifier labels */
            for (NvDsMetaList *cl = obj->classifier_meta_list; cl; cl = cl->next) {
                NvDsClassifierMeta *cmeta = (NvDsClassifierMeta *) cl->data;

                for (NvDsMetaList *ll = cmeta->label_info_list; ll; ll = ll->next) {
                    NvDsLabelInfo *linfo = (NvDsLabelInfo *) ll->data;
                    g_print("    classifier result: %s (prob=%.2f)\n",
                            linfo->result_label, linfo->result_prob);
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

/* Attach the probe to the sink pad of nvdsosd */
GstPad *osd_sink_pad = gst_element_get_static_pad(nvdsosd, "sink");
gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                  my_probe, NULL, NULL);
gst_object_unref(osd_sink_pad);
```

---

## 7. How to Access Metadata — Python

DeepStream Python bindings mirror the C API almost 1-to-1.

```python
import pyds

def my_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Step 1: get the batch meta
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # Step 2: iterate frames
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        print(f"source_id={frame_meta.source_id}  "
              f"frame_num={frame_meta.frame_num}")

        # Step 3: iterate objects
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)

            print(f"  class_id={obj_meta.class_id}  "
                  f"label={obj_meta.obj_label}  "
                  f"track_id={obj_meta.object_id}  "
                  f"conf={obj_meta.confidence:.2f}")

            # Bounding box
            r = obj_meta.rect_params
            print(f"  bbox: left={r.left:.1f} top={r.top:.1f} "
                  f"w={r.width:.1f} h={r.height:.1f}")

            # Step 4: classifier results
            l_cls = obj_meta.classifier_meta_list
            while l_cls is not None:
                cls_meta = pyds.NvDsClassifierMeta.cast(l_cls.data)
                l_lbl = cls_meta.label_info_list
                while l_lbl is not None:
                    lbl = pyds.NvDsLabelInfo.cast(l_lbl.data)
                    print(f"    {lbl.result_label}  prob={lbl.result_prob:.2f}")
                    try:
                        l_lbl = l_lbl.next
                    except StopIteration:
                        break
                try:
                    l_cls = l_cls.next
                except StopIteration:
                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
```

---

## 8. Adding Custom / User Metadata

### At frame or object level (C)

```c
#include "nvdsmeta.h"

typedef struct {
    float  my_score;
    gchar  my_label[64];
} MyCustomData;

/* Copy callback — called when the GstBuffer is copied */
static gpointer copy_my_meta(gpointer data, gpointer user_data) {
    MyCustomData *src = (MyCustomData *) data;
    MyCustomData *dst = g_new0(MyCustomData, 1);
    *dst = *src;
    return dst;
}

/* Release callback — called when the GstBuffer is freed */
static void release_my_meta(gpointer data, gpointer user_data) {
    g_free(data);
}

/* Inside your pad probe, after you have frame_meta */
void attach_custom_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta)
{
    /* 1. Acquire a user meta slot from the pool */
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

    /* 2. Create and fill your data */
    MyCustomData *my_data = g_new0(MyCustomData, 1);
    my_data->my_score = 0.95f;
    g_strlcpy(my_data->my_label, "cool_event", sizeof(my_data->my_label));

    /* 3. Fill the user meta fields */
    user_meta->user_meta_data  = my_data;
    user_meta->meta_type       = NVDS_USER_FRAME_META_EXAMPLE; // your enum value
    user_meta->copy_func       = copy_my_meta;
    user_meta->release_func    = release_my_meta;

    /* 4. Add to the frame's user meta list */
    nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

/* Reading the custom meta back downstream */
void read_custom_meta(NvDsFrameMeta *frame_meta)
{
    for (NvDsMetaList *l = frame_meta->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um->meta_type == NVDS_USER_FRAME_META_EXAMPLE) {
            MyCustomData *d = (MyCustomData *) um->user_meta_data;
            g_print("custom score=%.2f label=%s\n", d->my_score, d->my_label);
        }
    }
}
```

---

## 9. Metadata Before nvstreammux

If your plugin sits **upstream** of `nvstreammux`, the batch-level structures do
not exist yet. Use `gst_buffer_add_nvds_meta()` instead:

```c
/* Upstream plugin: add metadata to a single-frame GstBuffer */
NvDsMeta *meta = gst_buffer_add_nvds_meta(buf, my_data,
                                           NULL,         // user_data
                                           copy_my_meta,
                                           release_my_meta);
meta->meta_type = MY_UPSTREAM_META_TYPE;

/* Also set the transform function so nvstreammux can convert it */
meta->gst_to_nvds_meta_transform_func  = my_transform_func;
meta->gst_to_nvds_meta_release_func    = my_transform_release_func;
```

After `nvstreammux` processes the buffer, the meta is converted into an
`NvDsUserMeta` node and placed in `frame_meta->frame_user_meta_list`. Downstream
plugins can find it by searching that list for `MY_UPSTREAM_META_TYPE`.

Reference app: `sources/apps/sample_apps/deepstream-gst-metadata-test/`

---

## 10. Key Header Files

| Header | Location | Purpose |
|---|---|---|
| `gstnvdsmeta.h` | `sources/include/` | `gst_buffer_get_nvds_batch_meta()`, `gst_buffer_add_nvds_meta()` |
| `nvdsmeta.h` | `sources/include/` | All `NvDs*` struct definitions, pool APIs |
| `nvll_osd_struct.h` | `sources/include/` | `NvOSD_RectParams`, `NvOSD_TextParams`, `NvOSD_MaskParams` |
| `nvds_dewarper_meta.h` | `sources/include/` | Dewarper-specific metadata |
| `nvdsinfer.h` | `sources/include/` | Tensor meta for raw inference output |

---

## 11. Important Rules & Gotchas

| Rule | Detail |
|---|---|
| Always acquire from pools | Never `malloc` an `NvDsObjectMeta` directly. Use `nvds_acquire_*_from_pool()` functions so DeepStream can manage memory. |
| `rect_params` is the safe bbox | `detector_bbox_info` can have coordinates outside the frame. `rect_params` is always clipped. |
| `object_id` = `UNTRACKED_OBJECT_ID` | If no tracker (`nvtracker`) is in the pipeline, `object_id` is this sentinel value, not a meaningful ID. |
| `tracker_confidence` | Only NvDCF sets a real value. IOU, NvSORT, NvDeepSORT return `1.0`. |
| `rect_params` deprecation | Will be removed in a future release. Prefer `detector_bbox_info` / `tracker_bbox_info` for new code, but keep `rect_params` for OSD drawing for now. |
| `bInferDone` | Check this flag before reading object lists; it tells you inference actually ran. |
| Thread safety | Metadata is not thread-safe. Access it only inside pad probes or GStreamer element processing callbacks, not from arbitrary threads. |
| Python `StopIteration` | Python bindings raise `StopIteration` when a linked list is exhausted, not return `None`. Wrap `l = l.next` in `try/except StopIteration`. |

---

## Quick Reference — Most-Used API Functions

```c
/* Extraction */
NvDsBatchMeta *gst_buffer_get_nvds_batch_meta(GstBuffer *buf);

/* Pool allocation */
NvDsUserMeta     *nvds_acquire_user_meta_from_pool(NvDsBatchMeta *);
NvDsDisplayMeta  *nvds_acquire_display_meta_from_pool(NvDsBatchMeta *);
NvDsObjectMeta   *nvds_acquire_obj_meta_from_pool(NvDsBatchMeta *);

/* Attachment helpers */
void nvds_add_user_meta_to_batch(NvDsBatchMeta *, NvDsUserMeta *);
void nvds_add_user_meta_to_frame(NvDsFrameMeta *, NvDsUserMeta *);
void nvds_add_user_meta_to_obj(NvDsObjectMeta *, NvDsUserMeta *);
void nvds_add_display_meta_to_frame(NvDsFrameMeta *, NvDsDisplayMeta *);

/* Object list helpers */
void nvds_add_obj_meta_to_frame(NvDsFrameMeta *, NvDsObjectMeta *,
                                NvDsObjectMeta *parent_obj /* or NULL */);
void nvds_remove_obj_meta_from_frame(NvDsFrameMeta *, NvDsObjectMeta *);

/* Upstream (pre-mux) */
NvDsMeta *gst_buffer_add_nvds_meta(GstBuffer *, gpointer data,
                                   gpointer user_data,
                                   NvDsMetaCopyFunc copy_func,
                                   NvDsMetaReleaseFunc free_func);
```

---

*Source: NVIDIA DeepStream SDK Developer Guide — Metadata section*  
*Reference sample apps: `deepstream-test1` through `deepstream-test5`,*  
*`deepstream-user-metadata-test`, `deepstream-gst-metadata-test`*
