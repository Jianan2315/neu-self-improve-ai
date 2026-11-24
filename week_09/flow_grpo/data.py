"""Read a small, seeded training mixture before allocating the GPU."""
import random


def training_questions(settings, benchmark_questions):
    from datasets import load_dataset

    normalize = lambda text: " ".join(text.casefold().split())
    excluded = {normalize(q) for q in benchmark_questions}
    rows = []
    for source in settings["data_sources"]:
        count = source["questions"]
        if not count:
            continue
        stream = load_dataset(
            source["dataset"], revision=source["revision"], split="train",
            streaming=True, columns=["question", source["answer_column"]],
        ).shuffle(seed=settings["seed"], buffer_size=settings["shuffle_buffer"])
        taken = 0
        for item in stream:
            question = item["question"].strip()
            key = normalize(question)
            if key in excluded:
                continue
            answers = item[source["answer_column"]]
            if isinstance(answers, str):
                answers = [answers]
            rows.append(dict(source=source["dataset"], question=question,
                             reference_answers=list(answers)))
            excluded.add(key)
            taken += 1
            if taken == count:
                break
        if taken != count:
            raise ValueError(f"Not enough training questions in {source['dataset']}.")
    random.Random(settings["seed"]).shuffle(rows)
    return rows


def training_batches(settings, benchmark_questions):
    """Read disjoint batches once; each step keeps the requested source mixture."""
    expanded = dict(settings, data_sources=[dict(source, questions=source["questions"] * settings["training_steps"])
                                            for source in settings["data_sources"]])
    rows = training_questions(expanded, benchmark_questions)
    by_source = {source["dataset"]: [row for row in rows if row["source"] == source["dataset"]]
                 for source in settings["data_sources"]}
    batches = []
    for step in range(settings["training_steps"]):
        batch = []
        for source in settings["data_sources"]:
            count = source["questions"]
            batch.extend(by_source[source["dataset"]][step * count:(step + 1) * count])
        random.Random(settings["seed"] + step).shuffle(batch)
        batches.append(batch)
    return batches
