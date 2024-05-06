#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Window
{
public:
	Window();
	~Window();

	bool ShouldClose() const;
	void BeginFrame() const;
	void EndFrame() const;

	GLFWwindow* GetWindowHandle() const;

private:
	GLFWwindow* m_Window;
};