#pragma once

#include <algorithm>
#include <cstddef>

using atomicno_t = int;

static const std::size_t NUM_FORMULA_ELEMENTS = 8;
static const int FORMULA_ORDER[NUM_FORMULA_ELEMENTS] = {1, 6, 7, 8, 9, 15, 16, 17};

struct formula_t {
	std::size_t counts[NUM_FORMULA_ELEMENTS];

	formula_t() {
		std::fill_n(counts, NUM_FORMULA_ELEMENTS, std::size_t(0));
	}

	std::size_t atom_count() const {
		std::size_t total = 0;
		for (std::size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
			total += counts[i];
		}
		return total;
	}

	formula_t operator+(const formula_t &other) const {
		formula_t out;
		for (std::size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
			out.counts[i] = counts[i] + other.counts[i];
		}
		return out;
	}

	formula_t &operator+=(const formula_t &other) {
		for (std::size_t i = 0; i < NUM_FORMULA_ELEMENTS; ++i) {
			counts[i] += other.counts[i];
		}
		return *this;
	}
};
