#pragma once

#include <vector>
#include <imgui/imgui.h>

class CostPlot
{
public:
	void Update(float cost);
	void Draw();

private:
	std::vector<float> m_Costs;

	void Sparkline(const char* id, const float* values, int count, float min_v, float max_v, int offset, const ImVec4& col, const ImVec2& size);
};