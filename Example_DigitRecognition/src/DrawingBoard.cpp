#include "DrawingBoard.h"
#include "Config.h"
#include <iostream>

#define QUAD_SIZE_PX 12
#define DEFAULT_COLOR {1,1,1}
#define SELECTED_COLOR {0,0,0}

#define GET_PIXEL_VERTEX_INDEX(i, j, k) (4 * ((i) * DRAWING_BOARD_SIZE + (j)) + (k))

static const char* VERTEX_SHADER =
"#version 330\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aCol;\n"
"\n"
"out vec4 Col;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(aPos, 1.0);\n"
"	Col = vec4(aCol, 1.0);\n"
"}\n"
;

static const char* FRAGMENT_SHADER =
"#version 330\n"
"layout (location = 0) out vec4 FragColor;\n"
"in vec4 Col;\n"
"\n"
"void main()\n"
"{\n"
"	FragColor = Col;\n"
"}\n"
;

DrawingBoard::DrawingBoard(GLFWwindow* window) : m_Window(window)
{
	std::cout << "Generating Drawing Board OpenGL Buffers..." << std::endl;

	glGenVertexArrays(1, &m_VertexArrayId);
	glBindVertexArray(m_VertexArrayId);

	glGenBuffers(1, &m_VertexBufferId);
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(QuadVertex) * DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE * 4, nullptr, GL_DYNAMIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), 0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), (const void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	size_t indicesSize = DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE * 6;
	GLuint* indices = new GLuint[indicesSize];

	size_t ind = 0;
	GLuint baseIndex = 0;

	for (size_t i = 0; i < DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE; i++)
	{
		indices[ind + 0] = baseIndex + 0;
		indices[ind + 1] = baseIndex + 1;
		indices[ind + 2] = baseIndex + 2;
		indices[ind + 3] = baseIndex + 0;
		indices[ind + 4] = baseIndex + 2;
		indices[ind + 5] = baseIndex + 3;

		ind += 6;
		baseIndex += 4;
	}

	glGenBuffers(1, &m_IndexBufferId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBufferId);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indicesSize, indices, GL_STATIC_DRAW);

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &VERTEX_SHADER, NULL);
	glCompileShader(vertexShader);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &FRAGMENT_SHADER, NULL);
	glCompileShader(fragmentShader);

	std::cout << "Compiling Drawing Board Shaders..." << std::endl;

	m_ShaderId = glCreateProgram();
	glAttachShader(m_ShaderId, vertexShader);
	glAttachShader(m_ShaderId, fragmentShader);
	glLinkProgram(m_ShaderId);
	glValidateProgram(m_ShaderId);
	glDetachShader(m_ShaderId, vertexShader);
	glDetachShader(m_ShaderId, fragmentShader);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	glUseProgram(m_ShaderId);

	m_Vertices = new QuadVertex[DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE * 4];
	m_Center = glm::vec3(WINDOW_WIDTH / 2 - DRAWING_BOARD_SIZE / 2 * QUAD_SIZE_PX, WINDOW_HEIGHT / 2 - DRAWING_BOARD_SIZE / 2 * QUAD_SIZE_PX, 0);

	for (size_t i = 0; i < DRAWING_BOARD_SIZE; i++)
	{
		for (size_t j = 0; j < DRAWING_BOARD_SIZE; j++)
		{
			glm::vec3 center(m_Center.x + j * QUAD_SIZE_PX + 0.5 * QUAD_SIZE_PX, m_Center.y + i * QUAD_SIZE_PX + 0.5 * QUAD_SIZE_PX, 0);

			m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 0)] = { PixelToNDC((int)center.x - QUAD_SIZE_PX / 2, (int)center.y - QUAD_SIZE_PX / 2), DEFAULT_COLOR };
			m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 1)] = { PixelToNDC((int)center.x + QUAD_SIZE_PX / 2, (int)center.y - QUAD_SIZE_PX / 2), DEFAULT_COLOR };
			m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 2)] = { PixelToNDC((int)center.x + QUAD_SIZE_PX / 2, (int)center.y + QUAD_SIZE_PX / 2), DEFAULT_COLOR };
			m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 3)] = { PixelToNDC((int)center.x - QUAD_SIZE_PX / 2, (int)center.y + QUAD_SIZE_PX / 2), DEFAULT_COLOR };

			m_Data.push_back(0);
		}
	}

	delete[] indices;

	std::cout << "Drawing initialized!" << std::endl;
}

DrawingBoard::~DrawingBoard()
{
	std::cout << "Deleteing drawing board..." << std::endl;

	delete[] m_Vertices;

	glDeleteVertexArrays(1, &m_VertexArrayId);
	glDeleteBuffers(1, &m_VertexBufferId);
	glDeleteBuffers(1, &m_IndexBufferId);
	glDeleteProgram(m_ShaderId);
}

void DrawingBoard::Update()
{
	if (glfwGetMouseButton(m_Window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		double mouseX, mouseY;
		glfwGetCursorPos(m_Window, &mouseX, &mouseY);

		SelectPixel((int)mouseX, (int)mouseY);
	}
}

void DrawingBoard::Draw()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferId);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(QuadVertex) * DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE * 4, (const void*)m_Vertices);

	glUseProgram(m_ShaderId);
	glBindVertexArray(m_VertexArrayId);
	glDrawElements(GL_TRIANGLES, DRAWING_BOARD_SIZE * DRAWING_BOARD_SIZE * 6, GL_UNSIGNED_INT, 0);
}

void DrawingBoard::Reset()
{
	for (size_t i = 0; i < DRAWING_BOARD_SIZE; i++)
	{
		for (size_t j = 0; j < DRAWING_BOARD_SIZE; j++)
		{
			m_Data[i * DRAWING_BOARD_SIZE + j] = 0;

			UpdateVertexColor(i, j);
		}
	}

	std::cout << "Drawing board has been reset!" << std::endl;
}

std::vector<float> DrawingBoard::GetData() const
{
	return m_Data;
}

void DrawingBoard::SetData(std::vector<float> data)
{
	m_Data = data;

	for (size_t i = 0; i < DRAWING_BOARD_SIZE; i++)
	{
		for (size_t j = 0; j < DRAWING_BOARD_SIZE; j++)
		{
			UpdateVertexColor(i, j);
		}
	}

	std::cout << "Set new data for drawing board!" << std::endl;
}

void DrawingBoard::SelectPixel(int x, int y)
{
	if (x >= m_Center.x && x <= m_Center.x + QUAD_SIZE_PX * DRAWING_BOARD_SIZE && y >= m_Center.y && y <= m_Center.y + QUAD_SIZE_PX * DRAWING_BOARD_SIZE)
	{
		int i = (int)floor((y - (int)m_Center.y) / QUAD_SIZE_PX);
		int j = (int)floor((x - (int)m_Center.x) / QUAD_SIZE_PX);

		m_Data[i * DRAWING_BOARD_SIZE + j] = 1;

		UpdateVertexColor(i, j);
	}
}

glm::vec3 DrawingBoard::PixelToNDC(int x, int y)
{
	return glm::vec3((float)x / (float)WINDOW_WIDTH * 2.0f - 1.0f, 1.0f - (float)y / (float)WINDOW_HEIGHT * 2.0f, 0);
}

void DrawingBoard::UpdateVertexColor(size_t i, size_t j)
{
	glm::vec3 color = DEFAULT_COLOR;

	if (m_Data[i * DRAWING_BOARD_SIZE + j] == 1)
	{
		color = SELECTED_COLOR;
	}

	m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 0)].color = color;
	m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 1)].color = color;
	m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 2)].color = color;
	m_Vertices[GET_PIXEL_VERTEX_INDEX(i, j, 3)].color = color;
}