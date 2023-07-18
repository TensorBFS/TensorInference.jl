# Journal of Open Source Software (JOSS) paper

This directory contains the source code to compile the paper submitted to the
[Journal of Open Source Software](https://joss.theoj.org/).

# Compilation

*TIP*: If using Neovim, use the following command to compile and open the paper:

```
:AsyncRun -silent ./compile.sh; xdg-open ./paper.pdf
```

---

## Troubleshooting

- `Cannot connect to the Docker daemon at unix:/var/run/docker.sock`

Solution:

```sh
systemctl start docker
```

(Source: <https://stackoverflow.com/a/46329423/1706778>)
