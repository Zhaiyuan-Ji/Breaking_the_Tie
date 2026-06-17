from pathlib import Path

from caslr.data import RouterRecord, extract_router_columns, load_router_table


def test_load_router_table_reads_questions_and_router_columns():
    records = load_router_table(Path("data/sample/sample_router_data.csv"))

    assert len(records) == 10
    assert isinstance(records[0], RouterRecord)
    assert records[0].question == "Solve 2+2."
    assert records[0].dataset == "Math"
    assert records[0].success["Qwen2.5-7B-Instruct"] == 1
    assert records[0].success["Qwen2.5-Coder-7B-Instruct"] == 0


def test_extract_router_columns_preserves_expert_order():
    columns = [
        "question",
        "dataset",
        "router_A",
        "notes",
        "router_B",
    ]

    experts = extract_router_columns(columns)

    assert experts == ["A", "B"]
