#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

struct QuadVertex
{
	glm::vec3 position;
	glm::vec3 color;
};

class DrawingBoard
{
public:
	DrawingBoard(GLFWwindow* window);
	~DrawingBoard();
	
	void Update();
	void Draw();
	void Reset();
	std::vector<float> GetData() const;
	void SetData(std::vector<float> data);
private:
	GLFWwindow* m_Window;
	GLuint m_VertexArrayId;
	GLuint m_VertexBufferId;
	GLuint m_IndexBufferId;
	GLuint m_ShaderId;
	QuadVertex* m_Vertices;
	glm::vec3 m_Center;
	std::vector<float> m_Data;

	void SelectPixel(int x, int y);
	glm::vec3 PixelToNDC(int x, int y);
	void UpdateVertexColor(size_t i, size_t j);
};