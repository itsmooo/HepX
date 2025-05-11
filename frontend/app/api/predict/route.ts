import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Parse the incoming request
    const data = await request.json();
    console.log('Received data:', data); // Debug log

    // Forward to Python backend
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    console.log('Backend response:', result); // Debug log

    if (!response.ok) {
      throw new Error(result.error || 'Failed to process prediction');
    }

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
