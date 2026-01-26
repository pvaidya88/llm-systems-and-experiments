INVENTORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_number": {"type": "integer"},
        "page_summary": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_partial": {"type": "boolean"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["is_partial", "warnings"],
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": ["string", "null"]},
                    "caption": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "caption", "description", "confidence"],
            },
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": ["string", "null"]},
                    "caption": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "chart_type": {"type": ["string", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "caption", "description", "chart_type", "confidence"],
            },
        },
    },
    "required": ["page_number", "page_summary", "keywords", "quality", "tables", "figures"],
}

TABLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object_id": {"type": "string"},
        "page_number": {"type": "integer"},
        "label": {"type": ["string", "null"]},
        "caption": {"type": ["string", "null"]},
        "units": {"type": ["string", "null"]},
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "unit": {"type": ["string", "null"]},
                },
                "required": ["name", "unit"],
            },
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]},
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "has_merged_headers": {"type": "boolean"},
                "is_partial": {"type": "boolean"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["has_merged_headers", "is_partial", "warnings"],
        },
    },
    "required": [
        "object_id",
        "page_number",
        "label",
        "caption",
        "units",
        "columns",
        "rows",
        "notes",
        "quality",
    ],
}

FIGURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object_id": {"type": "string"},
        "page_number": {"type": "integer"},
        "label": {"type": ["string", "null"]},
        "caption": {"type": ["string", "null"]},
        "figure_type": {
            "type": "string",
            "enum": ["chart", "diagram", "infographic", "photo", "other"],
        },
        "quality": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_partial": {"type": "boolean"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["is_partial", "warnings"],
        },
        "chart": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chart_type": {"type": ["string", "null"]},
                        "x_axis": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": ["string", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                            "required": ["label", "unit"],
                        },
                        "y_axis": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": ["string", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                            "required": ["label", "unit"],
                        },
                        "series": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "points": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "x": {"type": ["number", "null"]},
                                                "y": {"type": ["number", "null"]},
                                                "label": {"type": ["string", "null"]},
                                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                            },
                                            "required": ["x", "y", "label", "confidence"],
                                        },
                                    },
                                },
                                "required": ["name", "points"],
                            },
                        },
                        "extraction_warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "chart_type",
                        "x_axis",
                        "y_axis",
                        "series",
                        "extraction_warnings",
                    ],
                },
            ]
        },
        "key_takeaways": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "object_id",
        "page_number",
        "label",
        "caption",
        "figure_type",
        "quality",
        "chart",
        "key_takeaways",
    ],
}

QUERY_EXPANSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "entities": {"type": "array", "items": {"type": "string"}},
        "likely_objects": {
            "type": "array",
            "items": {"type": "string", "enum": ["table", "figure"]},
        },
        "time_ranges": {"type": "array", "items": {"type": "string"}},
        "units": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords", "entities", "likely_objects", "time_ranges", "units"],
}

RERANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ranked_object_ids": {"type": "array", "items": {"type": "string"}},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ranked_object_ids", "reasons", "notes"],
}

FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "used_objects": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": ["OK", "NOT_FOUND", "AMBIGUOUS"]},
    },
    "required": ["answer", "citations", "used_objects", "status"],
}
