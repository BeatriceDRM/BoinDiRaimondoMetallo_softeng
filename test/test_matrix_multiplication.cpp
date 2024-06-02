#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <stdexcept>


/**
 * This function is used in order to compute the transpose of a matrix, 
 * given the matrix, its number of rows and its number of columns.
 **/
std::vector<std::vector<int>> transposeMatrix(std::vector<std::vector<int>> &A, int rows, int cols){
    std::vector<std::vector<int>> Mat(cols, std::vector<int>(rows, 0));

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols ; j++){
            Mat[i][j] = A[j][i];
        }
    }
    return Mat;
}


/**
 * 
 * TestMultiplyMatrices
 * 
 * Given two initial matrices, this test checks whether or not the multiplication
 * of two matrices A and B gives back the expected result matrix C
 *
 * Cases of errors found:
 * Error 6: Result matrix contains a number bigger than 100!
 * Error 12: The number of rows in A is equal to the number of columns in B!
 * Error 14: The result matrix C has an even number of rows!
 * Error 20: Number of columns in matrix A is odd!
 * 
 **/

TEST(MatrixMultiplicationTest, TestMultiplyMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

/**
 * 
 * TestMultiplyIdentity1
 * 
 * This test checks whether or not the multiplication between a general matrix A and the identity matrix gives back
 * the original matrix A: A*I=A
 *
 * Cases of errors found:
 * Error 1: Element-wise multiplication of ones detected!
 * Error 10: A row in matrix A contains more than one '1'!
 * Error 11: Every row in matrix B contains at least one '0'!
 * Error 13: The first element of matrix A is equal to the first element of matrix B!
 * Error 14: The result matrix C has an even number of rows!
 * Error 20: Number of columns in matrix A is odd!
 *
 **/

TEST(MatrixMultiplicationTest, TestMultiplyIdentity1) {
    std::vector<std::vector<int>> A = {
        {1, 1, 1},
        {4, 5, 4}
    };
    std::vector<std::vector<int>> B = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(3,0));
    multiplyMatrices(A, B, C, 2,3,3);

    std::vector<std::vector<int>> expected = {
        {1, 1, 1},
        {4, 5, 4}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";

    
}



/**
 * 
 * TestMultiplyIdentity2
 * 
 * This test checks whether or not the multiplication between the identity matrix and a general square matrix A gives back
 * the original matrix A
 *
 * Cases of errors found:
 * Error 1: Element-wise multiplication of ones detected!
 * Error 10: A row in matrix A contains more than one '1'!
 * Error 11: Every row in matrix B contains at least one '0'!
 * Error 12: The number of rows in A is equal to the number of columns in B!
 * Error 13: The first element of matrix A is equal to the first element of matrix B!
 * Error 15: A row in matrix A is filled entirely with 5s!
 * Error 18: Matrix A is a square matrix!
 * Error 20: Number of columns in matrix A is odd!
 * 
 **/

TEST(MatrixMultiplicationTest, TestMultiplyIdentity2) {

    std::vector<std::vector<int>> A = {
        {1, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 1}
    };
    std::vector<std::vector<int>> B = {
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5}
    };
    
    std::vector<std::vector<int>> C(5, std::vector<int>(5,0));
    multiplyMatrices(A, B, C, 5, 5, 5);

    std::vector<std::vector<int>> expected = {
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


/**
 *
 * NullProduct
 *
 * This case test is used to check if the multiplication between a general matrix A and the null matrix
 * returns the null matrix
 *
 * Cases of errors found:
 * Error 3: Matrix A contains a negative number!
 * Error 8: Result matrix contains zero!
 * Error 11: Every row in matrix B contains at least one '0'!
 *
 **/

TEST(MatrixMultiplicationTest, NullProduct){

    std::vector<std::vector<int>> A = {
        {55, 1, 2, 5},
        {4, -2, 38, 1},
        {56, 1, 2, 90}
    };
    std::vector<std::vector<int>> B = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    std::vector<std::vector<int>> C(3, std::vector<int>(4,0));
    multiplyMatrices(A, B, C, 3, 4, 4);

    std::vector<std::vector<int>> expected = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


/**
 * 
 * ScalarMult
 * 
 * This test is made in order to verify that the multiplication between two matrices containing a single element 
 * gives back the desired result.
 * 
 * Cases of errors found:
 * Error 12: The number of rows in A is equal to the number of columns in B!
 * Error 18: Matrix A is a square matrix!
 * Error 20: Number of columns in matrix A is odd!
 *
**/

TEST(MatrixMultiplicationTest, ScalarMult){
    std::vector<std::vector<int>> A(1, std::vector<int>(1, 3));
    std::vector<std::vector<int>> B(1, std::vector<int>(1, 7));

    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 21));

    multiplyMatrices(A,B,C,1,1,1);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


/**
 * 
 * TransposeOfProduct
 * 
 * This test checks if the property of the transpose (A*B)' = B'*A' is verified.
 * 
 * Case of errors found:
 * Error 1: Element-wise multiplication of ones detected!
 * Error 7: Result matrix contains a number between 11 and 20!
 * Error 8: Result matrix contains zero!
 * Error 10: A row in matrix A contains more than one '1'!
 * Error 11: Every row in matrix B contains at least one '0'!
 * Error 12: The number of rows in A is equal to the number of columns in B!
 * Error 14: The result matrix C has an even number of rows!
 * Error 17: Result matrix C contains the number 17!
 * Error 18: Matrix A is a square matrix!
 *
 **/

TEST(MatrixMultiplicationTest, TransposeOfProduct){

    std::vector<std::vector<int>> A = {
        {2, 2, 2, 2},
        {4, 4, 4, 4},
        {8, 8, 8, 8},
        {1, 1, 1, 1}
    };

    std::vector<std::vector<int>> B = {
        {1, 0, 0, 1},
        {1, 0, 0, 0},
        {0, 0, 0, 1},
        {1, 0, 0, 1}
    };

    std::vector<std::vector<int>> At(4, std::vector<int>(4, 0));
    std::vector<std::vector<int>> Bt(4, std::vector<int>(4, 0));
    std::vector<std::vector<int>> C(4, std::vector<int>(4, 0));

    At = transposeMatrix(A, 4, 4);
    Bt = transposeMatrix(B, 4, 4);


    std::vector<std::vector<int>> R1(4, std::vector<int>(4, 0));
    std::vector<std::vector<int>> R2(4, std::vector<int>(4, 0));
    std::vector<std::vector<int>> expected(4, std::vector<int>(4, 0));


    multiplyMatrices(A, B, R1, 4, 4, 4);
    multiplyMatrices(Bt, At, expected, 4, 4, 4);

    C = transposeMatrix(R1, 4, 4);
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

/**
* 
* IdempotentM
* 
* This test is used in order to understand whether, given an idempotent matrix A, 
* the property A*A=A is respected.
*
* Cases of errors found:
* Error 3: Matrix A contains a negative number!
* Error 5: Matrix B contains a negative number!
* Error 7: Result matrix contains a number between 11 and 20!
* Error 12: The number of rows in A is equal to the number of columns in B!
* Error 13: The first element of matrix A is equal to the first element of matrix B!
* Error 14: The result matrix C has an even number of rows!
* Error 18: Matrix A is a square matrix!
*
**/

TEST(MatrixMultiplicationTest, IdempotentM){

    std::vector<std::vector<int>> A = {
        {2, -2},
        {1, -1}
    };
    std::vector<std::vector<int>> B = {
        {2, -2},
        {1, -1}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> expected = {
        {2, -2},
        {1, -1}
    };

    multiplyMatrices(A,B,C,2,2,2);
    ASSERT_EQ(C,expected) << "Matrix multiplication test failed! :(((()";

}


/**
* 
* OrthogonalMatrix
* 
* This test checks if an orthogonal matrix A respects the property : A*A' = I
*
* Cases of errors found:
* Error 1: Element-wise multiplication of ones detected!
* Error 3: Matrix A contains a negative number!
* Error 5: Matrix B contains a negative number!
* Error 7: Result matrix contains a number between 11 and 20!
* Error 8: Result matrix contains zero!
* Error 11: Every row in matrix B contains at least one '0'!
* Error 12: The number of rows in A is equal to the number of columns in B!
* Error 13: The first element of matrix A is equal to the first element of matrix B!
* Error 18: Matrix A is a square matrix!
* Error 20: Number of columns in matrix A is odd!
* 
**/

TEST(MatrixMultiplicationTest, OrthogonalMatrix){

    std::vector<std::vector<int>> A = {
        {1, 0, 0},
        {0, 0, -1},
        {0, 1, 0}
    };
    std::vector<std::vector<int>> B(3, std::vector<int>(3, 0));
    B = transposeMatrix(A,3,3);
    std::vector<std::vector<int>> C(3, std::vector<int>(3, 0));
    std::vector<std::vector<int>> expected = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    multiplyMatrices(A, B, C, 3, 3, 3);
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";


}

/**
* 
* TwoMatrixVectors
* 
* This test is done in order to verify whether or not the result matrix respects the correct dimensions.
*
* Cases of errors found:
* Error 1: Element-wise multiplication of ones detected!
* Error 4: Matrix B contains the number 3!
* Error 9: Result matrix contains the number 99!
* Error 12: The number of rows in A is equal to the number of columns in B!
* Error 13: The first element of matrix A is equal to the first element of matrix B!
* Error 16: Matrix B contains the number 6!
*
**/

TEST(MatrixMultiplicationTest, TwoMatrixVectors){

    std::vector<std::vector<int>> A = {
        {1, 2, 3, 4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {1},
        {2},
        {3},
        {4},
        {5},
        {6}
    };
    std::vector<std::vector<int>> C(1, std::vector<int>(1, 0));
    std::vector<std::vector<int>> expected(1, std::vector<int>(1, 91));

    multiplyMatrices(A, B, C, 1, 6, 1);
    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";


}

/**
*
* AssociativityProperty
* 
* This test aims at verifying whether the associativity property holds: (A*B)*C = A*(B*C)
* 
* Cases of errors found:
* Error 1: Element-wise multiplication of ones detected!
* Error 6: Result matrix contains a number bigger than 100!
* Error 8: Result matrix contains zero!
* Error 11: Every row in matrix B contains at least one '0'!
* Error 12: The number of rows in A is equal to the number of columns in B!
* Error 13: The first element of matrix A is equal to the first element of matrix B!
* Error 14: The result matrix C has an even number of rows!
* Error 18: Matrix A is a square matrix!
* 
**/

TEST(MatrixMultiplicationTest, AssociativityProperty){
    std::vector<std::vector<int>> A = {
        {1, 0},
        {0, 0}
    };
    std::vector<std::vector<int>> B = {
        {1, 78},
        {12, 2}
    };
    std::vector<std::vector<int>> A2 = {
        {13, 0},
        {0, 0}
    };
    // T1 is a temporary matrix where I store the product between the first two matrices
    std::vector<std::vector<int>> T1(2, std::vector<int>(2, 0));
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    // the expected matrix is obtained by doing A*(B*A2)
    std::vector<std::vector<int>> expected{
        {13, 0},
        {0, 0}
    };

    multiplyMatrices(A, B, T1, 2, 2, 2);
    multiplyMatrices(T1, A2, C, 2, 2, 2);   // C will contain (A*B)*A2

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}