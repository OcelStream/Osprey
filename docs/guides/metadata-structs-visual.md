# DeepStream Metadata — Visual Struct Reference

Quick visual diagrams of every important struct and how they connect.
Companion to [metadata-guide.md](metadata-guide.md).

---

## Master Map (one GstBuffer → everything inside)

```
┌────────────────────────────────────────────────────────────────────┐
│ GstBuffer                                                          │
│   └─ GstMeta (type = NVDS_BATCH_GST_META)                         │
│        └─ NvDsBatchMeta                                            │
│             │                                                      │
│             ├─ num_frames_in_batch  (guint64)                      │
│             ├─ batch_user_meta_list ─────────────► NvDsUserMeta [] │
│             │                                                      │
│             └─ frame_meta_list                                     │
│                  │                                                 │
│                  ├─ NvDsFrameMeta [frame 0]                        │
│                  │    ├─ source_id, frame_num, buf_pts             │
│                  │    ├─ ntp_timestamp, pipeline_w/h              │
│                  │    ├─ bInferDone                                │
│                  │    ├─ frame_user_meta_list ──► NvDsUserMeta []  │
│                  │    ├─ display_meta_list ──────► NvDsDisplayMeta │
│                  │    └─ obj_meta_list                             │
│                  │         ├─ NvDsObjectMeta [obj 0]              │
│                  │         │    ├─ class_id, obj_label            │
│                  │         │    ├─ object_id (tracker ID)         │
│                  │         │    ├─ confidence, tracker_confidence  │
│                  │         │    ├─ detector_bbox_info             │
│                  │         │    ├─ tracker_bbox_info              │
│                  │         │    ├─ rect_params  ◄── use for OSD   │
│                  │         │    ├─ mask_params  (segmentation)     │
│                  │         │    ├─ text_params  (OSD label)        │
│                  │         │    ├─ obj_user_meta_list ─► UserMeta  │
│                  │         │    └─ classifier_meta_list            │
│                  │         │         ├─ NvDsClassifierMeta [0]    │
│                  │         │         │    └─ label_info_list       │
│                  │         │         │         └─ NvDsLabelInfo    │
│                  │         │         │              result_label   │
│                  │         │         │              result_prob     │
│                  │         │         └─ NvDsClassifierMeta [1] …  │
│                  │         └─ NvDsObjectMeta [obj 1] …            │
│                  │                                                 │
│                  └─ NvDsFrameMeta [frame 1] …                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## NvDsBatchMeta — Field Map

```
NvDsBatchMeta
├── [pools — internal, do not malloc from these directly]
│    frame_meta_pool
│    obj_meta_pool
│    classifier_meta_pool
│    display_meta_pool
│    user_meta_pool
│    label_info_meta_pool
│
├── num_frames_in_batch   guint64   actual frames in this batch
├── max_frames_in_batch   guint64   max possible
│
├── frame_meta_list       GList*    ──► NvDsFrameMeta nodes
└── batch_user_meta_list  GList*    ──► NvDsUserMeta nodes
```

---

## NvDsFrameMeta — Field Map

```
NvDsFrameMeta
├── source_id             guint     stream / camera index
├── pad_index             guint     nvstreammux input pad
├── batch_id              guint     index within the current batch
├── frame_num             gint      sequential frame number from source
├── buf_pts               guint64   GStreamer PTS (nanoseconds)
├── ntp_timestamp         guint64   wall-clock time (nanoseconds)
├── pipeline_width        gint      frame pixel width
├── pipeline_height       gint      frame pixel height
├── num_surfaces_per_frame gint     1 normally; >1 for dewarped 360 cam
├── bInferDone            gboolean  TRUE after any inference ran
├── num_obj_meta_in_use   guint     object count
│
├── obj_meta_list         GList*    ──► NvDsObjectMeta nodes
├── frame_user_meta_list  GList*    ──► NvDsUserMeta nodes
└── display_meta_list     GList*    ──► NvDsDisplayMeta nodes
```

---

## NvDsObjectMeta — Field Map

```
NvDsObjectMeta
├── frame_meta            ptr       back-pointer to parent NvDsFrameMeta
├── unique_component_id   gint      which nvinfer created this object
│
├── class_id              gint      class index (0=car, 1=person, …)
├── obj_label[64]         char[]    class name string
├── object_id             guint64   tracker ID (UNTRACKED_OBJECT_ID if none)
│
├── confidence            gfloat    detector confidence [0.0–1.0]
├── tracker_confidence    gfloat    tracker conf (NvDCF) or 1.0
│
├── detector_bbox_info    NvDsComp_BboxInfo   raw detector box (may overflow frame)
│    └─ org_bbox_coords   NvOSD_RectParams
│         left, top, width, height (float, pixels)
│
├── tracker_bbox_info     NvDsComp_BboxInfo   tracker-smoothed box
│    └─ org_bbox_coords   NvOSD_RectParams
│
├── rect_params           NvOSD_RectParams    CLIPPED final box ← use this
│    ├─ left, top, width, height
│    ├─ border_width, border_color (RGBA)
│    └─ has_bg_color, bg_color
│
├── mask_params           NvOSD_MaskParams    segmentation float mask
│    ├─ data              float*  (width × height floats)
│    ├─ width, height     guint
│    └─ size              guint   bytes allocated
│
├── text_params           NvOSD_TextParams    OSD label text
│    ├─ display_text      char*
│    ├─ x_offset, y_offset
│    └─ font_params, text_bg_clr
│
├── obj_user_meta_list    GList*    ──► NvDsUserMeta nodes
└── classifier_meta_list  GList*    ──► NvDsClassifierMeta nodes
```

---

## NvDsClassifierMeta + NvDsLabelInfo — Field Map

```
NvDsClassifierMeta
├── unique_component_id   gint    which secondary nvinfer
├── num_labels            guint   number of label entries
└── label_info_list       GList*  ──► NvDsLabelInfo nodes

    NvDsLabelInfo
    ├── label_id          gint    index within this classifier
    ├── num_classes       guint   total classes this classifier knows
    ├── result_class_id   guint   winning class index
    ├── result_label[64]  char[]  winning class name string
    └── result_prob       gfloat  winning class confidence
```

---

## NvDsUserMeta — Field Map

```
NvDsUserMeta
├── meta_type         NvDsMetaType   your custom enum value (>= NVDS_START_USER_META)
├── user_meta_data    gpointer       pointer to your struct
├── copy_func         gpointer       called on buffer copy  — must deep-copy your data
└── release_func      gpointer       called on buffer free  — must free your data
```

---

## NvDsDisplayMeta — Field Map

```
NvDsDisplayMeta                     (attached to frame, drawn by nvdsosd)
├── num_rects                        how many entries in rect_params[] are valid
├── num_labels                       how many entries in text_params[] are valid
├── num_lines
├── num_arrows
├── num_circles
│
├── rect_params[MAX_ELEMENTS]        array of NvOSD_RectParams
├── text_params[MAX_ELEMENTS]        array of NvOSD_TextParams
├── line_params[MAX_ELEMENTS]        array of NvOSD_LineParams
├── arrow_params[MAX_ELEMENTS]       array of NvOSD_ArrowParams
└── circle_params[MAX_ELEMENTS]      array of NvOSD_CircleParams
```

---

## Bounding Box: Which One to Use?

```
          nvinfer runs
              │
              ▼
   detector_bbox_info  ← raw network output, NOT clipped to frame
              │
              │  (if nvtracker in pipeline)
              ▼
   tracker_bbox_info   ← tracker-smoothed, NOT clipped to frame
              │
              ▼
   rect_params         ← CLIPPED to frame boundary ← use for drawing & coords
```

---

## Metadata Type Enum Values (NvDsMetaType)

| Value | Who Uses It |
|---|---|
| `NVDS_BATCH_GST_META` | Attached to GstBuffer by nvstreammux |
| `NVDS_FRAME_GST_META` | Per-frame GstMeta (pre-mux) |
| `NVDS_GST_CUSTOM_META` | Your pre-mux custom meta |
| `NVDS_OBJ_GST_META` | Object meta (internal) |
| `NVDS_DISPLAY_GST_META` | Display meta |
| `NVDS_TENSOR_OUTPUT_META` | Raw tensor output from nvinfer |
| `NVDS_PREPROCESS_BATCH_META` | nvdspreprocess ROI meta |
| `NVDS_OPTICAL_FLOW_META` | nvof motion vector meta |
| `NVDS_TRACKER_PAST_FRAME_META` | nvtracker past-frame data |
| `NVDS_LATENCY_MEASUREMENT_META` | Pipeline latency data |
| `≥ NVDS_START_USER_META` | **Your custom types** — define above this |

---

## C Iteration Pattern (compact reference)

```c
NvDsBatchMeta *b = gst_buffer_get_nvds_batch_meta(buf);
for (NvDsMetaList *fl = b->frame_meta_list; fl; fl = fl->next) {
    NvDsFrameMeta *f = (NvDsFrameMeta *)fl->data;
    for (NvDsMetaList *ol = f->obj_meta_list; ol; ol = ol->next) {
        NvDsObjectMeta *o = (NvDsObjectMeta *)ol->data;
        for (NvDsMetaList *cl = o->classifier_meta_list; cl; cl = cl->next) {
            NvDsClassifierMeta *c = (NvDsClassifierMeta *)cl->data;
            for (NvDsMetaList *ll = c->label_info_list; ll; ll = ll->next) {
                NvDsLabelInfo *l = (NvDsLabelInfo *)ll->data;
                /* use l->result_label, l->result_prob */
            }
        }
    }
}
```

## Python Iteration Pattern (compact reference)

```python
batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
l_f = batch.frame_meta_list
while l_f:
    frame = pyds.NvDsFrameMeta.cast(l_f.data)
    l_o = frame.obj_meta_list
    while l_o:
        obj = pyds.NvDsObjectMeta.cast(l_o.data)
        l_c = obj.classifier_meta_list
        while l_c:
            cls = pyds.NvDsClassifierMeta.cast(l_c.data)
            l_l = cls.label_info_list
            while l_l:
                lbl = pyds.NvDsLabelInfo.cast(l_l.data)
                # use lbl.result_label, lbl.result_prob
                try: l_l = l_l.next
                except StopIteration: break
            try: l_c = l_c.next
            except StopIteration: break
        try: l_o = l_o.next
        except StopIteration: break
    try: l_f = l_f.next
    except StopIteration: break
```

---

*See [metadata-guide.md](metadata-guide.md) for full explanations and custom metadata examples.*
