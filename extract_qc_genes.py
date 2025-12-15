#!/usr/bin/env python3
import re
import csv
import os
import json
import argparse
from typing import Set, Tuple, Dict, List, Any


def parse_gtf_attributes(attributes: str) -> Dict[str, str]:
    attr_dict = {}
    for attr in attributes.split(';'):
        attr = attr.strip()
        if attr:
            match = re.match(r'(\w+)\s+"([^"]*)"', attr)
            if match:
                key, value = match.groups()
                attr_dict[key] = value
            else:
                parts = attr.split(' ', 1)
                if len(parts) == 2:
                    attr_dict[parts[0]] = parts[1].strip()
    return attr_dict


def extract_genes_from_gtf(gtf_file: str, species: str) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    mito_genes = set()
    ribo_genes = set()
    gene_to_biotype: Dict[str, str] = {}

    ribo_pattern = re.compile(r'^RP[SL]\d+|^RPLP\d+|^RPSA$', re.IGNORECASE)
    mito_chromosomes = ['MT', 'chrMT', 'chrM', 'M', 'mitochondrion_genome']

    print(f"Processing {species} GTF file: {gtf_file}")

    try:
        with open(gtf_file, 'r', encoding='utf-8') as f:
            line_count = 0
            gene_count = 0

            for line in f:
                line_count += 1
                if line.startswith('#'):
                    continue

                fields = line.rstrip('\n').split('\t')
                if len(fields) < 9:
                    continue

                seqname, source, feature, start, end, score, strand, frame, attributes = fields
                if feature.lower() != 'gene':
                    continue

                gene_count += 1
                if gene_count % 5000 == 0:
                    print(f"  Processed {gene_count} genes...")

                attr_dict = parse_gtf_attributes(attributes)

                gene_name = None
                for name_field in ['gene_name', 'gene_symbol', 'gene_id']:
                    if name_field in attr_dict and attr_dict[name_field]:
                        gene_name = attr_dict[name_field]
                        break
                if not gene_name:
                    continue

                biotype = None
                for bt_field in ['gene_type', 'gene_biotype', 'transcript_biotype', 'biotype']:
                    if bt_field in attr_dict and attr_dict[bt_field]:
                        biotype = attr_dict[bt_field]
                        break
                if biotype is None:
                    biotype = "NA"

                gene_to_biotype.setdefault(gene_name, biotype)

                if seqname in mito_chromosomes:
                    mito_genes.add(gene_name)

                if ribo_pattern.match(gene_name):
                    ribo_genes.add(gene_name)

            print(f"  Total lines processed: {line_count}")
            print(f"  Total genes processed: {gene_count}")
            print(f"  Found {len(mito_genes)} mitochondrial genes")
            print(f"  Found {len(ribo_genes)} ribosomal genes")

    except FileNotFoundError:
        print(f"Error: GTF file not found: {gtf_file}")
    except Exception as e:
        print(f"Error processing {gtf_file}: {str(e)}")

    return mito_genes, ribo_genes, gene_to_biotype


def load_gtf_set(config_path: str, set_id: str) -> List[Dict[str, Any]]:
    """config jsonから指定idのgtf_configsを取り出す"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "gtf_sets" not in cfg or not isinstance(cfg["gtf_sets"], list):
        raise ValueError("Invalid config: top-level 'gtf_sets' (list) is required")

    for s in cfg["gtf_sets"]:
        if str(s.get("id")) == str(set_id):
            gtf_configs = s.get("gtf_configs", [])
            if not isinstance(gtf_configs, list):
                raise ValueError(f"Invalid config: gtf_configs for id={set_id} must be a list")
            return gtf_configs

    available = [str(s.get("id")) for s in cfg["gtf_sets"]]
    raise ValueError(f"Set id '{set_id}' not found. Available ids: {available}")


def create_qc_gene_list(config_path: str, set_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    gtf_configs = load_gtf_set(config_path, set_id)

    all_genes = []

    for item in gtf_configs:
        gtf_file = item["gtf_file"]
        species = item["species"]
        taxid = item["taxid"]

        if not os.path.exists(gtf_file):
            print(f"Warning: GTF file not found: {gtf_file}")
            continue

        mito_genes, ribo_genes, gene_to_biotype = extract_genes_from_gtf(gtf_file, species)

        for gene in sorted(mito_genes):
            all_genes.append([species, taxid, gene, gene_to_biotype.get(gene, "NA"), 'mitochondrial'])

        for gene in sorted(ribo_genes):
            all_genes.append([species, taxid, gene, gene_to_biotype.get(gene, "NA"), 'ribosomal'])

    output_file = os.path.join(output_dir, f'qc_gene_list_id{set_id}.csv')
    print(f"\nWriting results to {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Species', 'TaxID', 'Gene_symbol', 'Gene_biotype', 'Group'])
        writer.writerows(all_genes)

    print(f"QC gene list created with {len(all_genes)} total genes (Config_ID={set_id})")
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="gtf_config.json", help="Path to gtf config json")
    parser.add_argument("--id", default="1", help="gtf_set id to use (e.g., 1)")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    args = parser.parse_args()

    create_qc_gene_list(args.config_path, args.id, args.output_dir)


if __name__ == "__main__":
    main()
