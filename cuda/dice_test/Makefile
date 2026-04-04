# Makefile for dice_test (parallel across apps)
#
# Examples:
#   make print_benchmarks
#   make -j8 test_dice_all
#   make -j8 test_gpu_all
#   make -j8 test_all          # safe: per-app dice then gpu, apps in parallel
#
# Single app:
#   make test_dice app=nn_cuda
#   make test_gpu  app=nn_cuda

# -----------------------------
# User-overridable variables
# -----------------------------
app          ?= nn_cuda
test         ?= test_dice     # or test_gpu
RUN_SCRIPT   ?= run

BENCH_DIR    := ../$(app)
CFG_DIR      := cfg
SW_DIR       ?= sw_20pe
SUMMARY_DIR  := result_summary/$(app)

# DICE compiler (used to auto-generate missing app.N artifacts)
DICE_ILP_DIR ?= /data/jwang710/dice-ilp
DICE_TOOL    ?= $(DICE_ILP_DIR)/DICETool.py
DICE_CFG     ?= $(DICE_ILP_DIR)/config/non_ilp_config.json
# 1: regenerate per-slot pptx/meta from PTX via DICETool (do not modify SW_DIR)


# 0: copy-only from SW_DIR (no DICETool invocation)
DICE_REGEN   ?= 0

# How many apps to run in parallel (only used if you run without -j at top)
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

# -----------------------------
# Config files for each test type
# -----------------------------
CFG_gpu      := $(CFG_DIR)/gpgpusim_gpu_rtx2060s.config
CFG_dice     := $(CFG_DIR)/gpgpusim_dice_rtx2060s.config

# Metadata & pptx (only for dice)
META_SRC     := $(SW_DIR)/$(app).1.sm_52.meta
PPTX_SRC     := $(SW_DIR)/$(app).1.sm_52.pptx

# -----------------------------
# Logs
# -----------------------------
NOW          := $(shell date +%Y%m%d_%H%M%S)
LOG_SUFFIX   := $(strip $(test))_$(NOW)
STD_LOGFILE  := $(LOG_SUFFIX).log
STDERR_LOGFILE := $(LOG_SUFFIX)_stderr.log

# -----------------------------
# Benchmark lists (robust, no $(shell) parsing traps)
# -----------------------------
BENCHMARKS_LST      ?= benchmarks.lst
BENCHMARKS_SPEC_LST ?= benchmarks_spec.lst

define NL


endef

# Read file and turn newlines into spaces so make sees a word list
ifneq ($(wildcard $(BENCHMARKS_LST)),)
  BENCHMARKS := $(strip $(subst $(NL), ,$(file <$(BENCHMARKS_LST))))
else
  $(error Missing $(BENCHMARKS_LST) in $(CURDIR))
endif

ifneq ($(wildcard $(BENCHMARKS_SPEC_LST)),)
  BENCHMARKS_SPEC := $(strip $(subst $(NL), ,$(file <$(BENCHMARKS_SPEC_LST))))
else
  BENCHMARKS_SPEC :=
endif

ifeq ($(strip $(BENCHMARKS)),)
  $(error $(BENCHMARKS_LST) is empty (or contains only whitespace))
endif

# -----------------------------
# Phony targets
# -----------------------------
.PHONY: all \
        print_benchmarks \
        test_all test_dice_all test_gpu_all test_specific \
        copy_config copy_meta build run collect timeline \
        test_dice test_gpu clean clean_all

all: test_all

# -----------------------------
# Debug
# -----------------------------
print_benchmarks:
	@echo "BENCHMARKS=$(BENCHMARKS)"
	@echo "BENCHMARKS_SPEC=$(BENCHMARKS_SPEC)"

# -----------------------------
# Explicit per-app targets (no pattern-rule weirdness)
# -----------------------------
define MAKE_TEST_RULES
.PHONY: test_dice.$(1) test_gpu.$(1) test_both.$(1)

test_dice.$(1):
	@$(MAKE) test_dice app=$(1)

test_gpu.$(1):
	@$(MAKE) test_gpu app=$(1)

# Safe: sequential within one app to avoid clobbering $(BENCH_DIR)
test_both.$(1):
	@$(MAKE) test_dice app=$(1)
	@$(MAKE) test_gpu  app=$(1)
endef

$(foreach b,$(BENCHMARKS),$(eval $(call MAKE_TEST_RULES,$(b))))
$(foreach b,$(BENCHMARKS_SPEC),$(eval $(call MAKE_TEST_RULES,$(b))))

# -----------------------------
# Batch targets
# NOTE: do NOT pass -j into sub-make; let top-level "make -jN" control jobserver
# If you run without -j, we fall back to JOBS by explicitly invoking a parallel make.
# -----------------------------
test_dice_all:
ifeq ($(filter -j%,$(MAKEFLAGS)),)
	@$(MAKE) -j$(JOBS) $(addprefix test_dice.,$(BENCHMARKS))
else
	@$(MAKE) $(addprefix test_dice.,$(BENCHMARKS))
endif

test_gpu_all:
ifeq ($(filter -j%,$(MAKEFLAGS)),)
	@$(MAKE) -j$(JOBS) $(addprefix test_gpu.,$(BENCHMARKS))
else
	@$(MAKE) $(addprefix test_gpu.,$(BENCHMARKS))
endif

test_specific:
ifeq ($(filter -j%,$(MAKEFLAGS)),)
	@$(MAKE) -j$(JOBS) $(addprefix test_dice.,$(BENCHMARKS_SPEC))
else
	@$(MAKE) $(addprefix test_dice.,$(BENCHMARKS_SPEC))
endif

# Safe "both": apps in parallel, but dice->gpu sequential per app
test_all:
ifeq ($(filter -j%,$(MAKEFLAGS)),)
	@$(MAKE) -j$(JOBS) $(addprefix test_both.,$(BENCHMARKS))
else
	@$(MAKE) $(addprefix test_both.,$(BENCHMARKS))
endif

# -----------------------------
# Core steps (single app)
# -----------------------------

# Copy the appropriate config into the benchmark dir
copy_config:
	@echo "==> Copying config for $(test) into $(BENCH_DIR)..."
ifeq ($(test),test_gpu)
	@cp $(CFG_gpu) $(BENCH_DIR)/gpgpusim.config
	@cp $(CFG_DIR)/gpuwattch_gtx480.xml $(BENCH_DIR)/gpuwattch_gtx480.xml
	@cp $(CFG_DIR)/config_fermi_islip.icnt $(BENCH_DIR)/config_fermi_islip.icnt
else
	@cp $(CFG_dice) $(BENCH_DIR)/gpgpusim.config
	@cp $(CFG_DIR)/config_turing_islip.icnt $(BENCH_DIR)/config_turing_islip.icnt
endif

# Copy metadata & pptx (only for dice)
copy_meta:
ifeq ($(test),test_dice)
ifeq ($(DICE_REGEN),0)
	@echo "==> Copying metadata and PPTX into $(BENCH_DIR)..."
	@test -d "$(SW_DIR)" || { echo "[ERROR] Missing SW_DIR: $(SW_DIR)"; exit 1; }
	@test -f "$(META_SRC)" || { echo "[ERROR] Missing metadata source: $(META_SRC)"; exit 1; }
	@test -f "$(PPTX_SRC)" || { echo "[ERROR] Missing PPTX source: $(PPTX_SRC)"; exit 1; }
	@cp $(META_SRC) $(BENCH_DIR)/
	@cp $(PPTX_SRC) $(BENCH_DIR)/
else
	@echo "==> Syncing metadata and PPTX into $(BENCH_DIR)..."
	@rm -f $(BENCH_DIR)/$(app).*.sm_52.meta $(BENCH_DIR)/$(app).*.sm_52.pptx
	@ptx_files=$$(ls $(BENCH_DIR)/$(app).*.sm_52.ptx 2>/dev/null | sort -V); \
	src_stems=""; \
	for ptx in $$ptx_files; do \
		[ -f "$$ptx" ] || continue; \
		stem=$${ptx##*/}; stem=$${stem%.ptx}; \
		if ! grep -Eq '^[[:space:]]*([.]visible[[:space:]]+)?[.]entry[[:space:]]' "$$ptx"; then \
			rm -f $(BENCH_DIR)/$$stem.meta $(BENCH_DIR)/$$stem.pptx; \
			continue; \
		fi; \
		echo "    Regenerating $$stem.{pptx,meta} from PTX"; \
		ptx_abs=$$(realpath "$$ptx"); \
		( cd $(DICE_ILP_DIR) && python3 $(DICE_TOOL) -opt -i "$$ptx_abs" --cfg_file $(DICE_CFG) ); \
		if [ ! -f $(BENCH_DIR)/$$stem.meta ] || [ ! -f $(BENCH_DIR)/$$stem.pptx ]; then \
			echo "[ERROR] DICE generation failed for $$stem"; \
			exit 1; \
		fi; \
		src_stems="$$src_stems $$stem"; \
	done; \
	if [ -z "$$src_stems" ]; then \
		for ptx in $$ptx_files; do \
			[ -f "$$ptx" ] || continue; \
			stem=$${ptx##*/}; stem=$${stem%.ptx}; \
			rm -f $(BENCH_DIR)/$$stem.meta $(BENCH_DIR)/$$stem.pptx; \
		done; \
		echo "    Pruned stale slots (no kernel entry PTX found)"; \
	else \
		set -- $$src_stems; first_src_stem=$$1; \
		combined_stem=$(app).all.sm_52; \
		combined_pptx=$(BENCH_DIR)/$$combined_stem.pptx; \
		combined_meta=$(BENCH_DIR)/$$combined_stem.meta; \
		head -n 4 $(BENCH_DIR)/$$first_src_stem.pptx > $$combined_pptx; \
		for src_stem in $$src_stems; do \
			tail -n +5 $(BENCH_DIR)/$$src_stem.pptx >> $$combined_pptx; \
			echo >> $$combined_pptx; \
		done; \
		head -n 1 $(BENCH_DIR)/$$first_src_stem.meta > $$combined_meta; \
		for src_stem in $$src_stems; do \
			tail -n +2 $(BENCH_DIR)/$$src_stem.meta >> $$combined_meta; \
			echo >> $$combined_meta; \
		done; \
		pptx_func_count=$$(grep -Ec '^[[:space:]]*[.]visible[[:space:]]+[.]entry[[:space:]]' $$combined_pptx); \
		meta_func_count=$$(grep -Ec '^[[:space:]]*FUNCTION[[:space:]]*=' $$combined_meta); \
		if [ "$$pptx_func_count" -ne "$$meta_func_count" ]; then \
			echo "[ERROR] Combined function count mismatch: pptx=$$pptx_func_count meta=$$meta_func_count"; \
			exit 1; \
		fi; \
		dup_funcs=$$(grep -E '^[[:space:]]*FUNCTION[[:space:]]*=' $$combined_meta | awk -F'[[:space:]]*=[[:space:]]*' '{print $$2}' | sort | uniq -d); \
		if [ -n "$$dup_funcs" ]; then \
			echo "[ERROR] Duplicate function(s) in combined metadata: $$dup_funcs"; \
			exit 1; \
		fi; \
		for ptx in $$ptx_files; do \
			[ -f "$$ptx" ] || continue; \
			stem=$${ptx##*/}; stem=$${stem%.ptx}; \
			cp $$combined_meta $(BENCH_DIR)/$$stem.meta; \
			cp $$combined_pptx $(BENCH_DIR)/$$stem.pptx; \
		done; \
		rm -f $$combined_meta $$combined_pptx; \
		echo "    Duplicated combined metadata/PPTX to all slots for $(app)"; \
	fi
endif
else
	@echo "==> Skipping metadata/pptx for GPU test"
endif

# Build the benchmark
build:
	@echo "==> Building $(app) in $(BENCH_DIR)..."
	@$(MAKE) clean -C $(BENCH_DIR)
	@$(MAKE) -C $(BENCH_DIR)

# Run the benchmark
run:
	@echo "==> Running $(app) ($(test))..."
	@if [ ! -f "$(BENCH_DIR)/$(RUN_SCRIPT)" ]; then \
		echo "[ERROR] Missing run script: $(BENCH_DIR)/$(RUN_SCRIPT)"; \
		exit 1; \
	fi
	-@cd $(BENCH_DIR) && . ./$(RUN_SCRIPT) >$(STD_LOGFILE) 2>$(STDERR_LOGFILE)
	@echo "    Stdout log: $(BENCH_DIR)/$(STD_LOGFILE)"
	@echo "    Stderr log: $(BENCH_DIR)/$(STDERR_LOGFILE)"
	@mkdir -p $(SUMMARY_DIR)
	@cp $(BENCH_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR)/$(STD_LOGFILE)
	@cp $(BENCH_DIR)/$(STDERR_LOGFILE) $(SUMMARY_DIR)/$(STDERR_LOGFILE)

# Collect data from the copied log
collect:
	@echo "==> Collecting data for $(app)..."
	@python3 collect_data.py $(SUMMARY_DIR)/$(STD_LOGFILE) $(SUMMARY_DIR) --$(test)

timeline:
	@echo "==> Generating timelines from logs in $(SUMMARY_DIR)..."
	@set -e; \
	found=0; \
	for log_file in $$(find "$(SUMMARY_DIR)" -type f -name "test_dice_*.log" ! -name "*_stderr.log" | sort); do \
		found=1; \
		echo "  $$log_file"; \
		python3 create_timeline.py "$$log_file"; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "No suitable log files found under $(SUMMARY_DIR)."; \
	fi

# Clean up configs, logs, builds, and summary dir for the given app
clean:
	@echo "==> Cleaning up for $(app)..."
	-@rm -f $(BENCH_DIR)/gpgpusim.config
	-@rm -f $(BENCH_DIR)/*.log
	-@rm -f $(BENCH_DIR)/$(app).*.sm_52.meta
	-@rm -f $(BENCH_DIR)/$(app).*.sm_52.pptx
	-@rm -rf $(BENCH_DIR)/build
	-@rm -rf $(RUNDIR)
	-@rm -rf $(SUMMARY_DIR)
	@echo "==> Clean complete."

clean_all:
	@for b in $(BENCHMARKS); do \
	  $(MAKE) clean app=$$b; \
	done

# -----------------------------
# Full sequences (single app)
# -----------------------------
test_dice:
	@stamp=$$(date +%Y%m%d_%H%M%S); \
	log_suffix=test_dice_$$stamp; \
	$(MAKE) test=test_dice LOG_SUFFIX=$$log_suffix copy_config; \
	$(MAKE) test=test_dice LOG_SUFFIX=$$log_suffix copy_meta; \
	$(MAKE) test=test_dice LOG_SUFFIX=$$log_suffix build; \
	$(MAKE) test=test_dice LOG_SUFFIX=$$log_suffix run; \
	$(MAKE) test=test_dice LOG_SUFFIX=$$log_suffix collect

test_gpu:
	@stamp=$$(date +%Y%m%d_%H%M%S); \
	log_suffix=test_gpu_$$stamp; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix copy_config; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix build; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix run; \
	$(MAKE) test=test_gpu LOG_SUFFIX=$$log_suffix collect
