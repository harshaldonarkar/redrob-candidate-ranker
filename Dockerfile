# Self-contained sandbox for the Redrob ranker (submission_spec.md section 10.5).
# The ranking step uses the Python standard library ONLY, so there is no
# `pip install` step — the image is just Python + the two scripts.
#
# Build:  docker build -t redrob-ranker .
# Run:    docker run --rm -v "$PWD:/data" redrob-ranker \
#             --candidates /data/candidates.jsonl --out /data/submission.csv
# (see README.md for the full recipe, including the <=100-row sample run)

FROM python:3.13-slim

WORKDIR /app

# Only the deliverable scripts are needed; everything else is build-ignored
# (.dockerignore) so the 487 MB candidates.jsonl never enters the build context.
COPY rank.py validate_submission.py ./

# Args after the image name are passed straight through to rank.py. The default
# CMD assumes candidates.jsonl is mounted at /data; override --candidates/--out
# to point elsewhere.
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
