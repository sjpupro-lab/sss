# ce_core — CE Cell deterministic generation engine.
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lm

# Engine sources (no external deps).
SRCS = ce_core.c ce_storage.c ce_search.c ce_engine.c \
       ce_denoise.c ce_decode.c ce_extend.c ce_gen.c \
       ce_storage_io.c ce_feed_image.c ce_image_wave_refine.c \
       slig_signal.c

# Ingest unit pulls in stb_image.h (vendored, public domain). We compile
# it with reduced warnings — stb_image is third-party and contains many
# benign warnings under -Wall -Wextra.
INGEST_SRC = ce_ingest.c
INGEST_OBJ = ce_ingest.o
INGEST_CFLAGS = -Wall -O2 -std=c11 -Wno-unused-parameter -Wno-unused-but-set-variable

OBJS = $(SRCS:.c=.o) $(INGEST_OBJ)

BUILD_DIR = build

TEST_BINS = $(BUILD_DIR)/test_core \
            $(BUILD_DIR)/test_storage \
            $(BUILD_DIR)/test_engine \
            $(BUILD_DIR)/test_denoise \
            $(BUILD_DIR)/test_extend \
            $(BUILD_DIR)/test_gen \
            $(BUILD_DIR)/test_ingest \
            $(BUILD_DIR)/test_decode_image_block \
            $(BUILD_DIR)/test_feed_image \
            $(BUILD_DIR)/test_image_wave_refine \
            $(BUILD_DIR)/test_slig_signal

TOOL_BINS = $(BUILD_DIR)/ce_ingest

.PHONY: all test tools clean
all: $(BUILD_DIR) $(TEST_BINS) $(TOOL_BINS)
tools: $(BUILD_DIR) $(TOOL_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# stb_image lives inside ce_ingest.c — relax warnings just for that TU.
$(INGEST_OBJ): $(INGEST_SRC)
	$(CC) $(INGEST_CFLAGS) -c $< -o $@

$(BUILD_DIR)/test_core: tests/test_core.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_storage: tests/test_storage.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_engine: tests/test_engine.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_denoise: tests/test_denoise.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_extend: tests/test_extend.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_gen: tests/test_gen.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_ingest: tests/test_ingest.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_decode_image_block: tests/test_decode_image_block.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_feed_image: tests/test_feed_image.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_image_wave_refine: tests/test_image_wave_refine.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_slig_signal: tests/test_slig_signal.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/ce_ingest: tools/ce_ingest.c $(OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

test: all
	@echo "=== running ce_core tests ==="
	@for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		$$t || exit 1; \
		echo ""; \
	done
	@echo "=== ALL TESTS PASSED ==="

clean:
	rm -rf $(BUILD_DIR) $(OBJS)
