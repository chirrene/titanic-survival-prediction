
FROM rocker/r-ver:4.4.3

WORKDIR /app

COPY r/install_packages.R .
RUN Rscript install_packages.R

COPY r/main.R .
COPY data/ ./data/

CMD ["Rscript", "main.R"]
