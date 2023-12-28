#include <iostream>
#include "Examples/XORExample.h"

int main()
{
	std::srand((unsigned int)std::time(0));

	XORExample ex = XORExample();
	ex.InitBackprop(0.1f, 25000);
	ex.Prepare();

	while (true)
	{
		ex.RunIteration();

		if (ex.IsFinished())
		{
			break;
		}
	}
}