#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <filesystem>
#include <imgui/imgui.h>
#include <imgui/implot.h>
#include <NeuralNetwork.h>
#include "Config.h"
#include "Window.h"
#include "DrawingBoard.h"

static void _Sparkline(const char* id, const float* values, int count, float min_v, float max_v, int offset, const ImVec4& col, const ImVec2& size);

static void training_main()
{
	Window window;

	NeuralNetwork nn({ 784, 448, 224, 112, 10 }, NNActivationFunction::Sigmoid);
	nn.RandomizeLayers(-1, 1);

	DrawingBoard board(window.GetWindowHandle());

	std::vector<float> costs;

	bool isTraining = false;

	while (!window.ShouldClose())
	{
		window.BeginFrame();

		if (!isTraining)
		{
			board.Update();
			board.Draw();

			ImGui::Begin("Control", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
			ImGui::SetWindowSize({ 225, 180 });
			ImGui::SetWindowPos({ 10, 10 });

			static int x = 0;
			ImGui::InputInt("Number", &x);
			x = x > 9 ? 9 : x < 0 ? 0 : x;

			static char nameBuffer[64];
			ImGui::InputText("Filename", nameBuffer, sizeof(nameBuffer));

			ImGui::Spacing();

			if (ImGui::Button("Save", { 100, 25 }))
			{
				auto data = board.GetData();

				char buffer[128];
				sprintf_s(buffer, sizeof(buffer), "mnist/%s.pbm", nameBuffer);

				std::ofstream outfile(buffer);

				outfile << "P1" << std::endl;
				outfile << "# " << x << std::endl;
				outfile << DRAWING_BOARD_SIZE << " " << DRAWING_BOARD_SIZE << std::endl;

				for (size_t i = 0; i < DRAWING_BOARD_SIZE; i++)
				{
					for (size_t j = 0; j < DRAWING_BOARD_SIZE; j++)
					{
						outfile << (int)data[i * DRAWING_BOARD_SIZE + j] << " ";
					}

					if (i < DRAWING_BOARD_SIZE - 1)
					{
						outfile << std::endl;
					}
				}

				outfile.close();
			}

			ImGui::Spacing();

			if (ImGui::Button("Reset", { 100, 25 }))
			{
				board.Reset();
			}

			ImGui::Spacing();

			if (ImGui::Button("Start", { 100, 25 }))
			{
				isTraining = true;

				std::cout << "Starting training..." << std::endl;
			}

			ImGui::End();

			// USED FOR FASTER IMAGE GENERATION
			/*static bool test_q = false;
			if (glfwGetKey(window.GetWindowHandle(), GLFW_KEY_Q) == GLFW_PRESS)
			{
				if (!test_q)
				{
					board.Reset();

					test_q = true;
				}
			}
			else
			{
				test_q = false;
			}

			static int test_n = 100;
			static bool test_e = false;
			if (glfwGetKey(window.GetWindowHandle(), GLFW_KEY_E) == GLFW_PRESS)
			{
				if (!test_e)
				{
					auto data = board.GetData();

					char buffer[128];
					sprintf_s(buffer, sizeof(buffer), "mnist/%d_%d.pbm", x, test_n);

					std::cout << "Successfully saved: " << test_n << std::endl;

					test_n++;

					std::ofstream outfile(buffer);

					outfile << "P1" << std::endl;
					outfile << "# " << x << std::endl;
					outfile << DRAWING_BOARD_SIZE << " " << DRAWING_BOARD_SIZE << std::endl;

					for (size_t i = 0; i < DRAWING_BOARD_SIZE; i++)
					{
						for (size_t j = 0; j < DRAWING_BOARD_SIZE; j++)
						{
							outfile << (int)data[i * DRAWING_BOARD_SIZE + j] << " ";
						}

						if (i < DRAWING_BOARD_SIZE - 1)
						{
							outfile << std::endl;
						}
					}

					outfile.close();

					test_e = true;
				}
			}
			else
			{
				test_e = false;
			}*/
		}
		else
		{
			static bool init = false;

			if (!init)
			{
				std::vector<std::vector<float>> inputs;
				std::vector<std::vector<float>> outputs;

				for (const auto& entry : std::filesystem::directory_iterator("mnist"))
				{
					if (entry.path().extension() != ".pbm")
					{
						continue;
					}

					std::ifstream file(entry.path());
					std::string line;
					size_t ind = 0;
					std::vector<float> inp;

					while (std::getline(file, line))
					{
						if (line[0] == '#')
						{
							int val = line[2] - '0';
							std::vector<float> d;

							for (size_t i = 0; i < 10; i++)
							{
								d.push_back((int)i == val ? 1.0f : 0.0f);
							}

							outputs.push_back(d);
						}

						if (ind > 2)
						{
							for (size_t i = 0; i < line.size(); i++)
							{
								if (line[i] != ' ')
								{
									inp.push_back((float)(line[i] - '0'));
								}
							}
						}

						ind++;
					}

					inputs.push_back(inp);

					file.close();
				}

				nn.SetTrainingData(inputs, outputs);

				if (TRAINING_BATCH_SIZE > 0)
				{
					nn.SetStochastic(TRAINING_BATCH_SIZE);
				}

				init = true;
			}

			if (costs.size() < TRAINING_ITERATIONS)
			{
				nn.Train_Backpropagation();
				nn.Learn(TRAINING_LEARNING_RATE);

				costs.push_back(nn.CalculateCost());
			}
			else
			{
				static bool isSaved = false;

				if (!isSaved)
				{
					nn.SaveToFile("mnist/data.nn");

					isSaved = true;
				}
			}

			ImGui::Begin("Info", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
			ImGui::SetWindowSize({ 500, 175 });
			ImGui::SetWindowPos({ 10, 10 });

			const ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;

			if (ImGui::BeginTable("##table", 3, flags, ImVec2(500, 100)))
			{
				ImGui::TableSetupColumn("Epoch", ImGuiTableColumnFlags_WidthFixed, 100);
				ImGui::TableSetupColumn("Cost", ImGuiTableColumnFlags_WidthFixed, 100);
				ImGui::TableSetupColumn("Plot");
				ImGui::TableHeadersRow();
				ImPlot::PushColormap(ImPlotColormap_Cool);

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				ImGui::Text("%d", costs.size());
				ImGui::TableSetColumnIndex(1);
				ImGui::Text("%.5f", costs[costs.size() - 1]);
				ImGui::TableSetColumnIndex(2);
				ImGui::PushID(0);

				_Sparkline("##spark", costs.data(), (int)costs.size(), 0.0f, 2.5f, 0, ImPlot::GetColormapColor(0), ImVec2(-1, 100));

				ImGui::PopID();
				ImPlot::PopColormap();
				ImGui::EndTable();
			}

			ImGui::End();
		}

		window.EndFrame();
	}
}

static void runtime_main()
{
	Window window;

	NeuralNetwork nn("mnist/data.nn");
	DrawingBoard board(window.GetWindowHandle());

	int currentNumber = -1;

	while (!window.ShouldClose())
	{
		window.BeginFrame();

		board.Update();
		board.Draw();

		ImGui::Begin("Neural Network", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
		ImGui::SetWindowSize({ 125, 125 });
		ImGui::SetWindowPos({ 10, 10 });

		if (ImGui::Button("Reset"))
		{
			board.Reset();
		}

		if (ImGui::Button("Validate"))
		{
			std::vector<float> data = nn.Feedforward(board.GetData());
			float max = 0;

			std::cout << "> New validation" << std::endl;
			std::cout << std::fixed;
			std::cout << std::setprecision(3);

			for (size_t i = 0; i < data.size(); i++)
			{
				std::cout << i << "  -  " << data[i] << std::endl;

				if (data[i] > max)
				{
					max = data[i];
					currentNumber = (int)i;
				}
			}

			std::cout << std::endl;
		}

		ImGui::Text("Number: %d", currentNumber);

		ImGui::End();

		window.EndFrame();
	}
}

int main()
{
	if (TRAINING_BUILD)
	{
		training_main();
	}
	else
	{
		runtime_main();
	}
}

static void _Sparkline(const char* id, const float* values, int count, float min_v, float max_v, int offset, const ImVec4& col, const ImVec2& size)
{
	ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));

	if (ImPlot::BeginPlot(id, size, ImPlotFlags_CanvasOnly))
	{
		ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
		ImPlot::SetupAxesLimits(0, count - 1, min_v, max_v, ImGuiCond_Always);
		ImPlot::SetNextLineStyle(col);
		ImPlot::SetNextFillStyle(col, 0.25);
		ImPlot::PlotLine(id, values, count, 1, 0, ImPlotLineFlags_Shaded, offset);
		ImPlot::EndPlot();
	}

	ImPlot::PopStyleVar();
}