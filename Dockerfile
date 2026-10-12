# 1. Use the 'slim' image (Smallest official Python)
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files (saves tiny space)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set work directory
WORKDIR /app

# 4. Install dependencies
# We REMOVED apt-get entirely. Streamlit/Pillow usually run fine without it.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy app code (The .dockerignore file will filter junk out)
COPY . .

# 6. Expose port and run
EXPOSE 8080
CMD ["streamlit", "run", "AI_copycheck_withAPI.py", "--server.port=8080", "--server.address=0.0.0.0"]