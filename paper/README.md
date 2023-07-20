# Journal of Open Source Software (JOSS) paper

This directory contains the source code required to compile the paper submitted
to the [Journal of Open Source Software](https://joss.theoj.org/).

## Compilation

Make sure to have Docker installed. To compile the paper, navigate to the paper
directory and execute the compile.sh script using the following command:

```
./compile.sh
```

### Personal note (MRV)

To compile and open the paper with Neovim, run:

```
:AsyncRun -silent ./compile.sh; xdg-open ./paper.pdf
```

Or map the command a above to a keybinding, e.g.:

```
:lua vim.keymap.set("n", "<Leader>e", ":AsyncRun -silent ./compile.sh; xdg-open ./paper.pdf<CR>")
```

---

## Troubleshooting

- `Cannot connect to the Docker daemon at unix:/var/run/docker.sock`

Solution:

```sh
systemctl start docker
```

(Source: <https://stackoverflow.com/a/46329423/1706778>)
