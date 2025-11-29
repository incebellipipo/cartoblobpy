FROM incebellipipo/devcontainer:jammy

# Copy python package dependencies
COPY requirements* /tmp/

# Install python package dependencies
RUN pip install `find /tmp -maxdepth 1 -name 're*' -printf '-r %p '`