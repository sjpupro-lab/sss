CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -Iinclude -std=c11
LDFLAGS = -lm

SRC_DIR   = src
INC_DIR   = include
TEST_DIR  = tests
BUILD_DIR = build

SRCS = $(SRC_DIR)/spatial_grid.c \
       $(SRC_DIR)/spatial_morpheme.c \
       $(SRC_DIR)/spatial_layers.c \
       $(SRC_DIR)/spatial_match.c \
       $(SRC_DIR)/spatial_keyframe.c \
       $(SRC_DIR)/spatial_context.c \
       $(SRC_DIR)/spatial_generate.c \
       $(SRC_DIR)/spatial_io.c \
       $(SRC_DIR)/spatial_canvas.c \
       $(SRC_DIR)/spatial_subtitle.c

OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

TESTS = test_grid test_morpheme test_layers test_match test_keyframe test_context test_integration test_io test_cascade test_canvas test_adaptive test_subtitle

.PHONY: all clean test gpu_train_help

all: $(BUILD_DIR) $(OBJS)
	@echo "Build complete."

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
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

gpu_train_help:
	@echo "Kaggle GPU trainer (experimental):"
	@echo "  pip install -r requirements-gpu.txt"
	@echo "  python tools/kaggle_gpu_train.py --input data/sample_en.txt --max-clauses 50000 --checkpoint-every 5000"
	@echo "  output: build/gpu_models/gpu_model_final.pt"

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
