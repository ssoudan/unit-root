FROM mcr.microsoft.com/devcontainers/rust:bullseye

####################################
USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential lldb curl vim \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/.zshrc /home/vscode/.zshrc
RUN chown vscode:vscode /home/vscode/.zshrc

####################################
USER vscode

ENV PATH="/home/vscode/.cargo/bin:${PATH}"

RUN rustup default nightly

RUN rustup component add clippy rustfmt rust-src rust-analysis rls 

RUN cargo install cargo-deny eza cargo-criterion
