#include <libtensor/core/immutable.h>
#include "../test_utils.h"

using namespace libtensor;

namespace {

class immut : public immutable {
protected:
    virtual void on_set_immutable() { }
};

}

int main() {

    immut im;
    if(im.is_immutable()) {
        return fail_test("immutable_test::perform()", __FILE__, __LINE__,
            "New object must be mutable");
    }
    im.set_immutable();
    if(!im.is_immutable()) {
        return fail_test("immutable_test::perform()", __FILE__, __LINE__,
            "set_immutable() failed");
    }

    return 0;
}

