# The DDQC

mito and ribo gene preparation
1. specify which version of gtf to use in `gtf_config.json`

```
{
  "gtf_sets": [
    {
      "id": "1",
      "gtf_configs": [
        { "gtf_file": "/home/kariyayama/CellIO/Reference/gencode.v49/gencode.v49.primary_assembly.annotation.gtf", "species": "Human", "taxid": "9606" },
        { "gtf_file": "/home/kariyayama/CellIO/Reference/GRCm39/gencode.vM38.primary_assembly.annotation.gtf", "species": "Mouse", "taxid": "10090" },
        { "gtf_file": "/home/kariyayama/CellIO/Reference/GRCr8/Rattus_norvegicus.GRCr8.115.chr.gtf", "species": "Rat", "taxid": "10116" }
      ]
    }
  ]
}
```

2. run `extract_qc_genes.py` to get `qc_gene_list_id{}.csv`

```sh
conda activate cell-io-venv # set up via /data01/iharuto/cell-io-mappingctl/0_VENV_SETUP.sh or https://github.com/bioinfo-tsukuba/cell-io-mappingctl/blob/main/0_VENV_SETUP.sh
python3 extract_qc_genes.py # default config id is 1, for more, python3 extract_qc_genes.py --id 2
```