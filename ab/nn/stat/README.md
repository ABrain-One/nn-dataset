## Helpful Info 

## Quick Search with 'ripgrep' For Linux

Installation:

```bash
sudo snap install ripgrep --classic
```

Examples:

```bash
rg -F "\"accuracy\": 0.7655"  -g '1.json'
rg -F '"accuracy": 0.77' -g '**cifar-10**nnweb**/1.json'
```
