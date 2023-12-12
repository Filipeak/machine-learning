/**
* RESOURCES:
*  - 3Blue1Brown Explanation: https://www.3blue1brown.com/topics/neural-networks
*  - Example Series: https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw
*
* IDEAS:
*  - GFX - Neural Network Visualization & Includes checkup (Glad, GLFW, Imgui)
*  - Exporting / Importing
*  - Drawing Example
*/


#include <iostream>
#include "GFX/Window.h"
#include "GFX/CostPlot.h"
#include "GFX/NeuralNetworkVisualization.h"
#include "Examples/XORExample.h"

#define ENABLE_GFX 1

int main()
{
	std::srand((unsigned int)std::time(0));

#if ENABLE_GFX
	Window::Init();
#endif

	XORExample ex = XORExample();
	ex.InitBackprop(0.1f, 25000);
	ex.Prepare();

#if ENABLE_GFX
	CostPlot costPlot = CostPlot();
	NeuralNetworkVisualization nnVisualization = NeuralNetworkVisualization();

	nnVisualization.Init();
#endif

#if ENABLE_GFX
	while (!Window::ShouldClose())
#else
	while (true)
#endif
	{
#if ENABLE_GFX
		Window::BeginFrame();
#endif

		ex.RunIteration();

#if ENABLE_GFX
		if (!ex.IsFinished())
		{
			costPlot.Update(ex.GetCost());
		}

		costPlot.Draw();
		nnVisualization.Update(ex.GetData());

		Window::EndFrame();
#endif
	}

#if ENABLE_GFX
	nnVisualization.Terminate();

	Window::Terminate();
#endif
}