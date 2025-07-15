import {
    CopilotRuntime,
    ExperimentalEmptyAdapter,
    copilotRuntimeNextJSAppRouterEndpoint,
  } from "@copilotkit/runtime";;
  import { NextRequest } from "next/server";
   
  // You can use any service adapter here for multi-agent support.
  const serviceAdapter = new ExperimentalEmptyAdapter();
   
  const runtime = new CopilotRuntime({
    remoteEndpoints: [
      // Our FastAPI endpoint URL
      { url: "http://localhost:8000/copilotkit_remote" },
    ],
  });
   
  export const POST = async (req: NextRequest) => {
    const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
      runtime,
      serviceAdapter,
      endpoint: "/api/copilotkit",
    });
   
    return handleRequest(req);
  };