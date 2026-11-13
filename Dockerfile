# Flavor Swap web app: the substitution service + project website.
# Only the processed outputs are shipped; raw data (FooDB is 3.5 GB) stays out of the image.
FROM python:3.12-slim
WORKDIR /app
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt
COPY src/substitution_model.py src/substitution_model.py
COPY data/processed/ data/processed/
COPY results/d3/ground_truth_pairs.csv results/d3/ground_truth_pairs.csv
COPY webapp/ webapp/
ENV PORT=8000
EXPOSE 8000
# one worker keeps a single copy of the models in memory; threads handle concurrent requests
CMD gunicorn --chdir webapp app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
