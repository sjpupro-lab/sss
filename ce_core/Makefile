# ce_core — CE Cell deterministic generation engine.
CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lm

SRCS = ce_core.c ce_storage.c ce_search.c ce_engine.c \
       ce_denoise.c ce_decode.c ce_extend.c ce_gen.c

OBJS = $(SRCS:.c=.o)

BUILD_DIR = build

TEST_BINS = $(BUILD_DIR)/test_core \
            $(BUILD_DIR)/test_storage \
            $(BUILD_DIR)/test_engine \
            $(BUILD_DIR)/test_denoise \
            $(BUILD_DIR)/test_extend \
            $(BUILD_DIR)/test_gen

.PHONY: all test clean
all: $(BUILD_DIR) $(TEST_BINS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

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
