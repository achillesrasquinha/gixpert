### Installation

#### Installation via pip

The recommended way to install **gixpert** is via `pip`.

```shell
$ pip install gixpert
```

For instructions on installing python and pip see “The Hitchhiker’s Guide to Python” 
[Installation Guides](https://docs.python-guide.org/starting/installation/).

#### Building from source

`gixpert` is actively developed on [https://github.com](https://github.com/achillesrasquinha/gixpert)
and is always avaliable.

You can clone the base repository with git as follows:

```shell
$ git clone https://github.com/achillesrasquinha/gixpert
```

Optionally, you could download the tarball or zipball as follows:

##### For Linux Users

```shell
$ curl -OL https://github.com/achillesrasquinha/tarball/gixpert
```

##### For Windows Users

```shell
$ curl -OL https://github.com/achillesrasquinha/zipball/gixpert
```

Install necessary dependencies

```shell
$ cd gixpert
$ pip install -r requirements.txt
```

Then, go ahead and install gixpert in your site-packages as follows:

```shell
$ python setup.py install
```

Check to see if you’ve installed gixpert correctly.

```shell
$ gixpert --help
```