#include "letter_test.h"

namespace libtensor {

void letter_test::perform() throw(libtest::test_exception) {
	letter i, j, k, l;
	letter &i_ref(i);

	if(i != i) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != i");
	}
	if(!(i == i)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == i");
	}
	if(i != i_ref) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != i_ref");
	}
	if(!(i == i_ref)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == i_ref");
	}
	if(!(i != j)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i != j");
	}
	if(i == j) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed comparison operator: i == j");
	}

	i|j|k;

	if(!(i|j|k).contains(i)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(i)");
	}
	if((i|j|k).index_of(i) != 0) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(i)");
	}
	if(!(i|j|k).contains(j)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(j)");
	}
	if((i|j|k).index_of(j) != 1) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(j)");
	}
	if(!(i|j|k).contains(k)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(k)");
	}
	if((i|j|k).index_of(k) != 2) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index determination: (i|j|k).index_of(k)");
	}
	if((i|j|k).contains(l)) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed index location: (i|j|k).contains(l)");
	}

	if((i|j|k).letter_at(0) != i) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #1");
	}
	if((i|j|k).letter_at(0) == j) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #2");
	}
	if((i|j|k).letter_at(0) == k) {
		fail_test("letter_test::perform()", __FILE__, __LINE__,
			"Failed letter location: (i|j|k).letter_at(0) #2");
	}
}

} // namespace libtensor

