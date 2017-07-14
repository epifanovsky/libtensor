set(TESTS
    linalg_add_i_i_x_x_test
    linalg_copy_ij_ji_test
    linalg_mul2_i_i_i_x_test
    linalg_mul2_i_ip_p_x_test
    linalg_mul2_i_ipq_qp_x_test
    linalg_mul2_i_i_x_test
    linalg_mul2_ij_i_j_x_test
    linalg_mul2_ij_ip_jp_x_test
    linalg_mul2_ij_ip_pj_x_test
    linalg_mul2_ij_pi_jp_x_test
    linalg_mul2_ij_pi_pj_x_test
    linalg_mul2_i_pi_p_x_test
    linalg_mul2_x_p_p_test
    linalg_mul2_x_pq_pq_test
    linalg_mul2_x_pq_qp_test
)

libtensor_add_tests(linalg ${TESTS})

