#pragma once

#include <GLFW/glfw3.h>

class Window
{
public:
	static bool Init();
	static bool ShouldClose();
	static void BeginFrame();
	static void EndFrame();
	static void Terminate();

private:
	static GLFWwindow* m_Window;
};