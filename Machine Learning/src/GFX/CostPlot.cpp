#include "CostPlot.h"
#include <imgui/implot.h>

void CostPlot::Update(float cost)
{
	m_Costs.push_back(cost);
}

void CostPlot::Draw()
{
	ImGui::Begin("Plots");

	static ImGuiTableFlags flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable;

	if (ImGui::BeginTable("##table", 3, flags, ImVec2(900, 100)))
	{
		ImGui::TableSetupColumn("Iteration", ImGuiTableColumnFlags_WidthFixed, 100);
		ImGui::TableSetupColumn("Cost", ImGuiTableColumnFlags_WidthFixed, 100);
		ImGui::TableSetupColumn("Plot");
		ImGui::TableHeadersRow();
		ImPlot::PushColormap(ImPlotColormap_Cool);

		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::Text("%d", m_Costs.size());
		ImGui::TableSetColumnIndex(1);
		ImGui::Text("%.5f", m_Costs[m_Costs.size() - 1]);
		ImGui::TableSetColumnIndex(2);
		ImGui::PushID(0);

		Sparkline("##spark", m_Costs.data(), (int)m_Costs.size(), 0.0f, 0.25f, 0, ImPlot::GetColormapColor(0), ImVec2(-1, 100));

		ImGui::PopID();
		ImPlot::PopColormap();
		ImGui::EndTable();
	}

	ImGui::End();
}

void CostPlot::Sparkline(const char* id, const float* values, int count, float min_v, float max_v, int offset, const ImVec4& col, const ImVec2& size)
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