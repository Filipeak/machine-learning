/**
* RESOURCES:
*  - 3Blue1Brown Explanation: https://www.3blue1brown.com/topics/neural-networks
*  - Example Series: https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw
*
* IDEAS:
*  - GFX - Neural Network Visualization & Includes checkup (Glad, GLFW, Imgui)
*  - Stochastic Gradient Descent
*  - Exporting / Importing
*  - Drawing Example
*  - Optimization: Pointers & References checkup
*/


#include <iostream>
#include "GFX/Window.h"
#include "GFX/CostPlot.h"
#include "GFX/NeuralNetworkVisualization.h"
#include "Examples/XORExample.h"

int main()
{
	std::srand((unsigned int)std::time(0));

	Window::Init();

	auto ex = XORExample();
	auto costPlot = CostPlot();
	auto nnVisualization = NeuralNetworkVisualization();

	ex.InitBackprop(0.1f, 20000);
	ex.Prepare();
	nnVisualization.Init();

	while (!Window::ShouldClose())
	{
		Window::BeginFrame();

		ex.RunIteration();

		if (!ex.IsFinished())
		{
			costPlot.Update(ex.GetCost());
		}

		costPlot.Draw();
		nnVisualization.Update(ex.GetData());

		Window::EndFrame();
	}

	nnVisualization.Terminate();

	Window::Terminate();
}