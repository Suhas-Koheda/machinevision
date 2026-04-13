def construct_scene_graph(detections, semantic_tags):
    """
    Build relationships based on proximity and labels.
    """
    relationships = []
    
    # Map track_id to semantic tag
    tag_map = {t["track_id"]: t["semantic"] for t in semantic_tags if t["track_id"] != -1}
    
    for i, det1 in enumerate(detections):
        tid1 = det1.get("track_id", -1)
        label1 = det1["label"]
        semantic1 = tag_map.get(tid1, label1)
        
        for j, det2 in enumerate(detections):
            if i == j: continue
            tid2 = det2.get("track_id", -1)
            label2 = det2["label"]
            
            # Simple proximity check (distance between bounding box centers)
            c1 = [(det1["bbox"][0] + det1["bbox"][2])/2, (det1["bbox"][1] + det1["bbox"][3])/2]
            c2 = [(det2["bbox"][0] + det2["bbox"][2])/2, (det2["bbox"][1] + det2["bbox"][3])/2]
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            
            if dist < 150: # Threshold for 'near'
                rel = "near"
                if "person" in label1:
                    if "laptop" in label2: rel = "using"
                    elif "chair" in label2: rel = "sitting on"
                
                relationships.append({
                    "subject": f"{label1}#{tid1}",
                    "predicate": rel,
                    "object": f"{label2}#{tid2}"
                })
                
    return relationships
