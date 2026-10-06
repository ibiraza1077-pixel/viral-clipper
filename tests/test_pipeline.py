from clipper import pipeline


def test_new_job_is_saved_and_loadable(output_dir):
    job = pipeline.new_job("https://example.com/v", {"num_clips": 3})
    assert (output_dir / job["id"] / "job.json").exists()
    assert pipeline.load(job["id"]) == job
    assert pipeline.load("missing") is None


def test_list_jobs_newest_first_and_skips_corrupt(output_dir):
    a = pipeline.new_job("a", {})
    b = pipeline.new_job("b", {})
    b["created"] = a["created"] + 10
    pipeline.save(b)
    bad = output_dir / "broken"
    bad.mkdir()
    (bad / "job.json").write_text("{not json")
    assert [j["source"] for j in pipeline.list_jobs()] == ["b", "a"]


def test_mark_interrupted_only_touches_unfinished(output_dir):
    running = pipeline.new_job("r", {})
    running["status"] = "rendering"
    pipeline.save(running)
    done = pipeline.new_job("d", {})
    done["status"] = "done"
    pipeline.save(done)

    pipeline.mark_interrupted()

    assert pipeline.load(running["id"])["status"] == "error"
    assert "Interrupted" in pipeline.load(running["id"])["error"]
    assert pipeline.load(done["id"])["status"] == "done"


def test_slug():
    assert pipeline._slug("Why I QUIT my $100k job!") == "why-i-quit-my-100k-job"
    assert pipeline._slug("🔥🔥") == "clip"
