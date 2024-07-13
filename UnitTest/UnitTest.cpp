#include "pch.h"
#include "CppUnitTest.h"
#include "../Matrix.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest
{
	TEST_CLASS(MatrixTest)
	{
	public:
		
		TEST_METHOD(MatrixInit)
		{
			const int rows = 2;
			const int cols = 3;

			Matrix<rows, cols> m1;

			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					Assert::AreEqual(m1.getElement(i, j), 0.0);
				}
			}
		}
	};
}
