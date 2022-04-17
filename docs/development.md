# Development

## Prepare environment

Create a virtual environment

```shell
python3 -m venv .venv
```

Switch to virtual environment (not required when running from Visual Studio Code)

```shell
. .venv/bin/activate
```

## Build C sources

Build and install sources

```shell
python setup.py develop
```

or

```shell
pip install -e .
```

## Prepare for distribution

Freeze dependencies:

```shell
pip3 freeze | grep -v cybervision > requirements.txt
```

Install frozen dependencies:

```shell
pip3 install -r requirements.txt
```
