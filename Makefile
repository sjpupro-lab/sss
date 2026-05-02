CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -Iinclude -Ice_core -std=c11
LDFLAGS = -lm

SRC_DIR     = src
INC_DIR     = include
TEST_DIR    = tests
BUILD_DIR   = build
CE_CORE_DIR = ce_core

SRCS = $(SRC_DIR)/spatial_grid.c \
       $(SRC_DIR)/spatial_morpheme.c \
       $(SRC_DIR)/spatial_layers.c \
       $(SRC_DIR)/spatial_match.c \
       $(SRC_DIR)/spatial_keyframe.c \
       $(SRC_DIR)/spatial_context.c \
       $(SRC_DIR)/spatial_generate.c \
       $(SRC_DIR)/spatial_io.c \
       $(SRC_DIR)/spatial_canvas.c \
       $(SRC_DIR)/spatial_subtitle.c \
       $(SRC_DIR)/spatial_recluster.c \
       $(SRC_DIR)/spatial_image.c \
       $(SRC_DIR)/spatial_image_gen.c

# ce_core library — CEUnit base + slig_signal v2.
# ce_ingest.c is intentionally excluded (depends on a vendored
# third_party/stb_image.h that ships separately and is not needed
# by the SSS integration path; SSS uses its own PPM reader).
CE_CORE_SRCS = $(CE_CORE_DIR)/ce_core.c \
               $(CE_CORE_DIR)/ce_storage.c \
               $(CE_CORE_DIR)/ce_search.c \
               $(CE_CORE_DIR)/ce_engine.c \
               $(CE_CORE_DIR)/ce_denoise.c \
               $(CE_CORE_DIR)/ce_decode.c \
               $(CE_CORE_DIR)/ce_extend.c \
               $(CE_CORE_DIR)/ce_gen.c \
               $(CE_CORE_DIR)/ce_storage_io.c \
               $(CE_CORE_DIR)/ce_feed_image.c \
               $(CE_CORE_DIR)/ce_image_wave_refine.c \
               $(CE_CORE_DIR)/slig_signal.c \
               $(CE_CORE_DIR)/slig_codebook.c \
               $(CE_CORE_DIR)/slig_pipeline.c \
               $(CE_CORE_DIR)/slig_tick_math.c \
               $(CE_CORE_DIR)/slig_material_harmonic.c \
               $(CE_CORE_DIR)/ce_hybrid_vae.c \
               $(CE_CORE_DIR)/ce_masked_train.c \
               $(CE_CORE_DIR)/ce_residual_codebook.c \
               $(CE_CORE_DIR)/sss_rowvae.c \
               $(CE_CORE_DIR)/sss_io.c

CE_CORE_OBJS = $(patsubst $(CE_CORE_DIR)/%.c,$(BUILD_DIR)/ce_core_%.o,$(CE_CORE_SRCS))

OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS)) $(CE_CORE_OBJS)

TESTS = test_grid test_morpheme test_layers test_match test_keyframe test_context test_integration test_io test_cascade test_canvas test_adaptive test_subtitle test_recluster test_refine test_image_roundtrip test_image_gen test_tick_math test_material test_gen_routed

.PHONY: all clean test bench_context bench_refine image_tools

all: $(BUILD_DIR) $(OBJS)
	@echo "Build complete."

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ce_core objects are prefixed (ce_core_NAME.o) so they cannot collide
# with anything in src/. Includes both -Iinclude and -Ice_core so cross-
# module references compile cleanly.
$(BUILD_DIR)/ce_core_%.o: $(CE_CORE_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Test targets
test: $(addprefix $(BUILD_DIR)/,$(TESTS))
	@echo "=== Running all tests ==="
	@for t in $(TESTS); do \
		echo "--- $$t ---"; \
		./$(BUILD_DIR)/$$t || exit 1; \
		echo ""; \
	done
	@echo "=== ALL TESTS PASSED ==="

$(BUILD_DIR)/test_grid: $(TEST_DIR)/test_grid.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_morpheme: $(TEST_DIR)/test_morpheme.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_layers: $(TEST_DIR)/test_layers.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_match: $(TEST_DIR)/test_match.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_keyframe: $(TEST_DIR)/test_keyframe.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_context: $(TEST_DIR)/test_context.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_integration: $(TEST_DIR)/test_integration.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_io: $(TEST_DIR)/test_io.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_cascade: $(TEST_DIR)/test_cascade.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_canvas: $(TEST_DIR)/test_canvas.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_adaptive: $(TEST_DIR)/test_adaptive.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_subtitle: $(TEST_DIR)/test_subtitle.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_recluster: $(TEST_DIR)/test_recluster.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_refine: $(TEST_DIR)/test_refine.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_context: $(TEST_DIR)/bench_context.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_refine: $(TEST_DIR)/bench_refine.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_image_roundtrip: $(TEST_DIR)/test_image_roundtrip.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_image_gen: $(TEST_DIR)/test_image_gen.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_tick_math: $(TEST_DIR)/test_tick_math.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_material: $(TEST_DIR)/test_material.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_gen_routed: $(TEST_DIR)/test_gen_routed.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_v3: tools/bench_v3.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

bench_v3: $(BUILD_DIR)/bench_v3
	@./$(BUILD_DIR)/bench_v3

$(BUILD_DIR)/img2grid: tools/img2grid.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/grid2img: tools/grid2img.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/train_images_ce: tools/train_images_ce.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/gen_image_ce: tools/gen_image_ce.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

# Sculpt-based image generator (sss_rowvae engine).
$(BUILD_DIR)/sss_gen: tools/sss_gen.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

sss_gen: $(BUILD_DIR)/sss_gen
	@echo "Built sss_gen. Pipeline:"
	@echo "  python3 scripts/sss_train.py --labels data/sss_demo/labels.tsv \\"
	@echo "      --root data/sss_demo --out build/models/demo.sss --size 64"
	@echo "  ./build/sss_gen build/models/demo.sss \"red circle draw\" out.ppm 1 1.0"

$(BUILD_DIR)/make_demo_dataset: tools/make_demo_dataset.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

$(BUILD_DIR)/train_demo: tools/train_demo.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/verify_hybrid: tools/verify_hybrid.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

verify_hybrid: $(BUILD_DIR)/verify_hybrid
	@echo "Built verify_hybrid. Example:"
	@echo "  ./build/verify_hybrid data/demo/img/*.ppm"

# Single-cell wave residual visualizer — drops one CE Cell on an empty
# canvas and writes a PPM. Used to eyeball wave shape before trusting
# it inside the masked-train loop.
$(BUILD_DIR)/wave_debug: tools/wave_debug.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

wave_debug: $(BUILD_DIR)/wave_debug
	@echo "Built wave_debug. Examples:"
	@echo "  ./build/wave_debug ripple.ppm --dir 6 --freq 16 --speed 4 --tick 30"
	@echo "  ./build/wave_debug beam.ppm   --dir 7 --sigma 8000 --tick 40"
	@echo "  ./build/wave_debug horiz.ppm  --dir 0 --sigma 4000"

demo_tools: $(BUILD_DIR)/make_demo_dataset $(BUILD_DIR)/train_demo $(BUILD_DIR)/gen_image_ce $(BUILD_DIR)/verify_hybrid
	@echo "Built demo tools. Pipeline:"
	@echo "  ./build/make_demo_dataset data/demo"
	@echo "  ./build/train_demo data/demo build/models/demo"
	@echo "  ./build/gen_image_ce build/models/demo.ces \"red apple\" out.ppm 0 50 200"

train_images_ce: $(BUILD_DIR)/train_images_ce
	@echo "Built train_images_ce. Example:"
	@echo "  ./build/train_images_ce build/models/demo  data/img/*.ppm"

gen_image_ce: $(BUILD_DIR)/gen_image_ce
	@echo "Built gen_image_ce. Example:"
	@echo "  ./build/gen_image_ce build/synth/demo.ces \"red apple\" out.ppm 0 50 200"

image_tools: $(BUILD_DIR)/img2grid $(BUILD_DIR)/grid2img
	@echo "Built image prototype tools. Examples:"
	@echo "  ./build/img2grid  in.ppm  out.ppm"
	@echo "  ./build/grid2img  model.spai 0 kf0.ppm"

bench_context: $(BUILD_DIR)/bench_context
	@./$(BUILD_DIR)/bench_context

bench_refine: $(BUILD_DIR)/bench_refine
	@./$(BUILD_DIR)/bench_refine

# ─── Wikipedia integration test (manual run) ─────────────────
$(BUILD_DIR)/test_wiki: $(TEST_DIR)/test_wiki.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

wiki: $(BUILD_DIR)/test_wiki
	@echo "Built test_wiki. Run with:"
	@echo "  ./build/test_wiki data/sample_ko.txt"
	@echo "  ./build/test_wiki data/sample_en.txt"
	@echo "  ./build/test_wiki data/sample_en.txt 500"

# ─── Benchmarks: STS-B + Perplexity ──────────────────────────
$(BUILD_DIR)/bench_stsb: $(TEST_DIR)/bench_stsb.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_perplexity: $(TEST_DIR)/bench_perplexity.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_word_predict: $(TEST_DIR)/bench_word_predict.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/bench_qa: $(TEST_DIR)/bench_qa.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

bench_word: $(BUILD_DIR)/bench_word_predict
	@echo "Built bench_word_predict. Run with:"
	@echo "  ./build/bench_word_predict data/sample_ko.txt"
	@echo "  ./build/bench_word_predict data/sample_en.txt 1000"

bench_qa: $(BUILD_DIR)/bench_qa
	@echo "Built bench_qa. Run with:"
	@echo "  ./build/bench_qa data/qa.tsv"

# ─── Streaming trainer (line-by-line, no full-file buffering) ────
$(BUILD_DIR)/stream_train: tools/stream_train.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

# ─── Interactive chat REPL ───────────────────────────────────────
$(BUILD_DIR)/chat: tools/chat.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

chat: $(BUILD_DIR)/chat
	@echo "Built chat. Example:"
	@echo "  ./build/chat --load build/models/wiki5k.spai"
	@echo "  ./build/chat --train data/wiki5k.txt --max 5000"

stream: $(BUILD_DIR)/stream_train
	@echo "Built stream_train. Example:"
	@echo "  ./build/stream_train --input data/sample_en.txt --max 50000 \\"
	@echo "                       --save build/models/wiki50k.spai --verify"

bench: $(BUILD_DIR)/bench_stsb $(BUILD_DIR)/bench_perplexity \
       $(BUILD_DIR)/bench_word_predict $(BUILD_DIR)/bench_qa
	@echo "=== Benchmarks built ==="
	@echo "  ./build/bench_stsb          data/stsb.tsv"
	@echo "  ./build/bench_perplexity    data/sample_ko.txt"
	@echo "  ./build/bench_word_predict  data/sample_ko.txt"
	@echo "  ./build/bench_qa            data/qa.tsv"
	@echo ""
	@if [ -f data/stsb.tsv ]; then \
		echo "=== Running bench_stsb ==="; \
		./$(BUILD_DIR)/bench_stsb data/stsb.tsv || true; \
	else \
		echo "(skip bench_stsb: data/stsb.tsv not found)"; \
	fi
	@echo ""
	@if [ -f data/sample_ko.txt ]; then \
		echo "=== Running bench_perplexity on sample_ko.txt ==="; \
		./$(BUILD_DIR)/bench_perplexity data/sample_ko.txt 500 || true; \
		echo "=== Running bench_word_predict on sample_ko.txt ==="; \
		./$(BUILD_DIR)/bench_word_predict data/sample_ko.txt 500 || true; \
	elif [ -f data/sample_en.txt ]; then \
		echo "=== Running bench_perplexity on sample_en.txt ==="; \
		./$(BUILD_DIR)/bench_perplexity data/sample_en.txt 500 || true; \
		echo "=== Running bench_word_predict on sample_en.txt ==="; \
		./$(BUILD_DIR)/bench_word_predict data/sample_en.txt 500 || true; \
	else \
		echo "(skip bench_perplexity/word: no sample_*.txt)"; \
	fi
	@echo ""
	@if [ -f data/qa.tsv ]; then \
		echo "=== Running bench_qa ==="; \
		./$(BUILD_DIR)/bench_qa data/qa.tsv || true; \
	else \
		echo "(skip bench_qa: data/qa.tsv not found — run data/make_qa.ps1)"; \
	fi

clean:
	rm -rf $(BUILD_DIR)
