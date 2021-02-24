#include <libtensor/core/index_range.h>
#include "../test_utils.h"

using namespace libtensor;

int test_ctor() {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    index_range<2> ir(i1, i2);

    return 0;
}


int main() {
    return test_ctor();
}

