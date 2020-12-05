#include <libtensor/symmetry/block_labeling.h>
#include "../test_utils.h"

using namespace libtensor;


/** \test Tests assigning labels.
 **/
int test_basic_1() {

    static const char testname[] = "block_labeling_test::test_basic_1()";

    try {

        libtensor::index<4> i1, i2;
        i2[0] = 1; i2[1] = 1; i2[2] = 3; i2[3] = 3;
        dimensions<4> bidims(index_range<4>(i1, i2));
        block_labeling<4> el(bidims);

        mask<4> m0011, m0100;
        m0100[1] = true; m0011[2] = true; m0011[3] = true;

        // Check assignment
        el.assign(m0011, 0, 0);
        el.assign(m0011, 1, 2);
        el.assign(m0011, 2, 1);
        el.assign(m0011, 3, 3);
        el.assign(m0100, 0, 0);

        if (el.get_dim_type(1) == el.get_dim_type(2)) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Equal types for dims 1 and 2.");
        }
        if (el.get_dim_type(2) != el.get_dim_type(3)) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Non-equal types for dims 2 and 3.");
        }
        if (el.get_label(el.get_dim_type(1), 0) != 0) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 0).");
        }
        if (el.get_label(el.get_dim_type(1), 1) !=
                (block_labeling<4>::label_t) -1) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 1).");
        }
        if (el.get_label(el.get_dim_type(2), 0) != 0) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 0).");
        }
        if (el.get_label(el.get_dim_type(2), 1) != 2) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 1).");
        }
        if (el.get_label(el.get_dim_type(2), 2) != 1) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 2).");
        }
        if (el.get_label(el.get_dim_type(2), 3) != 3) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (2, 3).");
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}

/** \test Tests assigning labels and matching them
 **/
int test_basic_2() {

    static const char testname[] = "block_labeling_test::test_basic_2()";

    try {

        libtensor::index<4> i1, i2;
        i2[0] = 1; i2[1] = 1; i2[2] = 3; i2[3] = 3;
        dimensions<4> bidims(index_range<4>(i1, i2));
        block_labeling<4> el(bidims);

        mask<4> m0100, m0010, m0001;
        m0100[1] = true; m0010[2] = true; m0001[3] = true;

        // Check assignment
        for (block_labeling<4>::label_t i = 0; i < 4; i++)
            el.assign(m0001, i, i);
        el.assign(m0100, 0, 0); // ag
        el.assign(m0100, 1, 2); // au
        for (block_labeling<4>::label_t i = 0; i < 4; i++)
            el.assign(m0010, i, i);

        el.match();

        if (el.get_dim_type(1) == el.get_dim_type(2)) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Equal types for dims 1 and 2.");
        }
        if (el.get_dim_type(2) != el.get_dim_type(3)) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Non-equal types for dims 2 and 3.");
        }
        if (el.get_label(el.get_dim_type(1), 0) != 0) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 0).");
        }
        if (el.get_label(el.get_dim_type(1), 1) != 2) {
            return fail_test(testname, __FILE__, __LINE__,
                    "Unexpected label at (1, 1).");
        }
        for (block_labeling<4>::label_t i = 0; i < 4; i++) {
            if (el.get_label(el.get_dim_type(2), i) != i) {
                return fail_test(testname, __FILE__, __LINE__,
                        "Unexpected label in dim 2.");
            }
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}

/** \test Copy labeling (3-dim)
 **/
int test_copy_1() {

    static const char testname[] = "block_labeling_test::test_copy_1()";

    try {

        libtensor::index<3> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 5;
        dimensions<3> bidims(index_range<3>(i1, i2));
        block_labeling<3> elem(bidims);

        mask<3> m001, m110;
        m110[0] = true; m110[1] = true; m001[2] = true;


        // assign labels
        for (size_t i = 0; i < 4; i++) elem.assign(m110, i, i);
        elem.assign(m001, 0, 0); elem.assign(m001, 1, 1);
        elem.assign(m001, 2, 1); elem.assign(m001, 3, 2);
        elem.assign(m001, 4, 3); elem.assign(m001, 5, 3);

        block_labeling<3> elema(elem);

        mask<3> done;
        for (size_t i = 0; i < 3; i++) {

            if (done[i]) continue;

            size_t cur_type = elem.get_dim_type(i);
            size_t cur_type_a = elema.get_dim_type(i);

            if (elem.get_dim(cur_type) != elema.get_dim(cur_type_a))
                return fail_test(testname, __FILE__, __LINE__,
                        "Dimensions do not match.");

            for (size_t j = 0; j < elem.get_dim(cur_type); j++) {
                if (elem.get_label(cur_type, j) !=
                        elema.get_label(cur_type_a, j))
                    return fail_test(testname, __FILE__, __LINE__,
                            "Labels do not match.");
            }

            done[i] = true;

            for (size_t j = i + 1; j < 3; j++) {
                if (elem.get_dim_type(j) == cur_type) {
                    if (elema.get_dim_type(j) != cur_type_a)
                        return fail_test(testname, __FILE__, __LINE__,
                                "Wrong dim type.");

                    done[j] = true;
                }
            }
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Permutation of labeling (3-dim)
 **/
int test_permute_1() {

    static const char testname[] = "block_labeling_test::test_permute_1()";

    try {

        libtensor::index<3> i1, i2;
        i2[0] = 3; i2[1] = 3; i2[2] = 5;
        dimensions<3> bidims(index_range<3>(i1, i2));
        block_labeling<3> elem(bidims);

        mask<3> m001, m110;
        m110[0] = true; m110[1] = true; m001[2] = true;

        // assign labels
        for (size_t i = 0; i < 4; i++) elem.assign(m110, i, i);
        elem.assign(m001, 0, 0); elem.assign(m001, 1, 1);
        elem.assign(m001, 2, 1); elem.assign(m001, 3, 2);
        elem.assign(m001, 4, 3); elem.assign(m001, 5, 3);

        block_labeling<3> elema(elem), elemb(elem);

        permutation<3> pa, pb;
        pa.permute(0, 1);
        pb.permute(0, 1).permute(1, 2);

        // [012->120]
        // result: ijk before kij
        // before: ijk result jki
        elema.permute(pa);
        elemb.permute(pb);

        sequence<3, size_t> mapa, mapb;
        for (size_t i = 0; i < 3; i++) mapa[i] = mapb[i] = i;
        permutation<3> pinva(pa, true), pinvb(pb, true);

        pinva.apply(mapa);
        pinvb.apply(mapb);

        mask<3> done;
        for (size_t i = 0; i < 3; i++) {

            if (done[i]) continue;

            size_t cur_type = elem.get_dim_type(i);
            size_t cur_type_a = elema.get_dim_type(mapa[i]);
            size_t cur_type_b = elemb.get_dim_type(mapb[i]);

            if (elem.get_dim(cur_type) != elema.get_dim(cur_type_a))
                return fail_test(testname, __FILE__, __LINE__,
                        "Dimensions do not match (a)");
            if (elem.get_dim(cur_type) != elemb.get_dim(cur_type_b))
                return fail_test(testname, __FILE__, __LINE__,
                        "Dimensions do not match (b)");

            for (size_t j = 0; j < elem.get_dim(cur_type); j++) {
                if (elem.get_label(cur_type, j) !=
                        elema.get_label(cur_type_a, j))
                    return fail_test(testname, __FILE__, __LINE__,
                            "Labels do not match (a)");
                if (elem.get_label(cur_type, j) !=
                        elemb.get_label(cur_type_b, j))
                    return fail_test(testname, __FILE__, __LINE__,
                            "Labels do not match (b)");
            }

            done[i] = true;

            for (size_t j = i + 1; j < 3; j++) {
                if (elem.get_dim_type(j) == cur_type) {
                    if (elema.get_dim_type(mapa[j]) != cur_type_a)
                        return fail_test(testname, __FILE__, __LINE__,
                                "Wrong dim type (a)");
                    if (elema.get_dim_type(mapb[j]) != cur_type_b)
                        return fail_test(testname, __FILE__, __LINE__,
                                "Wrong dim type (b)");

                    done[j] = true;
                }
            }
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}

/** \test Transfer of labeling (1-dim to 3-dim)
 **/
int test_transfer_1() {

    static const char testname[] = "block_labeling_test::test_transfer_1()";

    try {

        libtensor::index<1> i1a, i2a;
        libtensor::index<3> i1b, i2b;
        i2a[0] = 5;
        i2b[0] = 3; i2b[1] = 3; i2b[2] = 5;
        dimensions<1> bidimsa(index_range<1>(i1a, i2a));
        dimensions<3> bidimsb(index_range<3>(i1b, i2b));

        mask<1> m; m[0] = true;
        mask<3> m001, m110; m110[0] = true; m110[1] = true; m001[2] = true;

        block_labeling<1> elema(bidimsa);
        block_labeling<3> elemb(bidimsb);

        // assign labels
        elema.assign(m, 0, 0); elema.assign(m, 1, 1);
        elema.assign(m, 2, 1); elema.assign(m, 3, 2);
        elema.assign(m, 4, 3); elema.assign(m, 5, 3);

        sequence<1, size_t> mapa; mapa[0] = 2;

        transfer_labeling(elema, mapa, elemb);

        size_t typea = elema.get_dim_type(0);
        size_t typeb = elemb.get_dim_type(mapa[0]);

        if (elema.get_dim(typea) != elemb.get_dim(typeb))
            return fail_test(testname, __FILE__, __LINE__,
                    "Dimensions do not match.");

        for (size_t j = 0; j < elema.get_dim(typea); j++) {
            if (elema.get_label(typea, j) !=
                    elemb.get_label(typeb, j))
                return fail_test(testname, __FILE__, __LINE__,
                        "Labels do not match.");
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}

/** \test Transfer of labelings(3-dim to 2-dim)
 **/
int test_transfer_2() {

    static const char testname[] = "block_labeling_test::test_transfer_2()";

    try {

        libtensor::index<3> i1a, i2a;
        libtensor::index<2> i1b, i2b;
        i2a[0] = 3; i2a[1] = 3; i2a[2] = 5;
        i2b[0] = 5; i2b[1] = 3;
        dimensions<3> bidimsa(index_range<3>(i1a, i2a));
        dimensions<2> bidimsb(index_range<2>(i1b, i2b));

        mask<3> m001, m110; m110[0] = true; m110[1] = true; m001[2] = true;
        mask<3> m01, m10; m10[0] = true; m01[1] = true;

        block_labeling<3> elema(bidimsa);
        block_labeling<2> elemb(bidimsb);

        // assign labels
        for (size_t i = 0; i < 4; i++) elema.assign(m110, i, i);
        elema.assign(m001, 0, 0); elema.assign(m001, 1, 1);
        elema.assign(m001, 2, 1); elema.assign(m001, 3, 2);
        elema.assign(m001, 4, 3); elema.assign(m001, 5, 3);

        sequence<3, size_t> mapa;
        mapa[0] = 1; mapa[1] = (size_t) -1; mapa[2] = 0;

        transfer_labeling(elema, mapa, elemb);

        mask<3> done;
        for (size_t i = 0; i < 3; i++) {
            if (mapa[i] == (size_t) -1) continue;

            size_t typea = elema.get_dim_type(i);
            size_t typeb = elemb.get_dim_type(mapa[i]);

            if (elema.get_dim(typea) != elemb.get_dim(typeb))
                return fail_test(testname, __FILE__, __LINE__,
                        "Dimensions do not match.");

            for (size_t j = 0; j < elema.get_dim(typea); j++) {
                if (elema.get_label(typea, j) !=
                        elemb.get_label(typeb, j))
                    return fail_test(testname, __FILE__, __LINE__,
                            "Labels do not match.");
            }
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_basic_1() |
    test_basic_2() |
    test_copy_1() |
    test_permute_1() |
    test_transfer_1() |
    test_transfer_2() |

    0;
}

