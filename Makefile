# Convenience targets for the supplied SLURM workflows.
CKPT ?= assets/nucaliby_v1.ckpt
OUT  ?= results
TAI_WEIGHT ?= 50
PROTPARDELLE_WEIGHTS ?= assets/protpardelle-1c

.PHONY: ensemble design-aa design-nt design-aa-ensemble design-nt-ensemble design-tai \
	design-motif-sira design-motif-fse design-motif-both design-vienna fold metrics metrics-paper train

ensemble: ## Protpardelle partial-denoising ensemble; pass PDB=...
	sbatch scripts/inference/generate_ensemble.sbatch --pdb $(PDB) \
		--weights-dir $(PROTPARDELLE_WEIGHTS) --out $(OUT)/ensembles

design-aa:       ## amino-acid design
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet aa --guide lcp --out $(OUT)/aa/designs.csv

design-nt:       ## nucleotide design
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --guide stop:weight=100 --out $(OUT)/nt/designs.csv

design-aa-ensemble: ## ensemble-conditioned amino-acid design; pass ENSEMBLE_DIR=...
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--ensemble-dir $(ENSEMBLE_DIR) --alphabet aa --guide lcp --out $(OUT)/aa_ensemble/designs.csv

design-nt-ensemble: ## ensemble-conditioned nucleotide design; pass ENSEMBLE_DIR=...
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--ensemble-dir $(ENSEMBLE_DIR) --alphabet nt --guide stop:weight=100 \
		--out $(OUT)/nt_ensemble/designs.csv

design-tai: ## tAI guidance; pass TAI_WEIGHT=... (default: 50)
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --guide stop:weight=100 --guide tai:weight=$(TAI_WEIGHT),organism=bl21_de3 \
		--out $(OUT)/tai_w$(TAI_WEIGHT)/designs.csv

design-motif-sira: ## SiRA motif experiment, 16 samples per backbone
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --num-seqs 16 --guide stop:weight=200 \
		--guide motif:file=assets/motifs/sira.txt,weight=50,beta=2 --out $(OUT)/motif_sira/designs.csv

design-motif-fse: ## HIV-1 FSE motif experiment, 16 samples per backbone
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --num-seqs 16 --guide stop:weight=200 \
		--guide motif:file=assets/motifs/hiv1_fse.txt,weight=50,beta=2 --out $(OUT)/motif_fse/designs.csv

design-motif-both: ## joint SiRA + HIV-1 FSE experiment, 16 samples per backbone
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --num-seqs 16 --guide stop:weight=200 \
		--guide motif:file=assets/motifs/sira.txt,weight=50,beta=2 \
		--guide motif:file=assets/motifs/hiv1_fse.txt,weight=50,beta=2 --out $(OUT)/motif_both/designs.csv

design-vienna: ## ViennaRNA ensemble-free-energy guidance experiment
	sbatch scripts/inference/design.sbatch --ckpt $(CKPT) --test-ids assets/test_pdb_ids.txt \
		--alphabet nt --guide stop:weight=100 --guide vienna:weight=10 --out $(OUT)/vienna_w10/designs.csv

fold:            ## ColabFold; pass ROW=aa or ROW=nt
	sbatch scripts/inference/fold.sbatch --designs $(OUT)/$(ROW)/designs.csv \
		--out $(OUT)/$(ROW)/fold --skip-existing

metrics:         ## score folded designs; pass ROW=...
	PYTHONPATH=. python scripts/inference/metrics.py --selected $(OUT)/$(ROW)/fold/selected.csv \
		--predictions $(OUT)/$(ROW)/fold/predictions --out $(OUT)/$(ROW)/metrics.csv

metrics-paper:   ## paper evaluation filter: shared 183 IDs, chain A
	PYTHONPATH=. python scripts/inference/metrics.py --selected $(OUT)/$(ROW)/fold/selected.csv \
		--predictions $(OUT)/$(ROW)/fold/predictions --ids assets/test_pdb_ids.txt --chains A \
		--out $(OUT)/$(ROW)/metrics.csv

train:           ## submit a training job
	sbatch scripts/training/train.sbatch --data-root $(DATA_ROOT) --out-dir runs/nucaliby
